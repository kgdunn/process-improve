"""The "Show the Python code" panels: the code each step of the page shows must do what the page did.

``web/python.js`` writes the code from the page's state. It is run here under Node, as the
page runs it, and the code it writes is run in Python, so the code a reader copies is checked
to reproduce what the page computed. A panel shows its step after the steps it builds on, and
each panel's code is run here on its own, in a fresh namespace, as a reader who copies only it
would run it.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import io
import json
import re

import numpy as np
import pytest

from process_improve.visualization.colors import colorscale
from tests.web._support import (
    ALL_DESIGNS,
    FACTORS,
    coded,
    correlated_study,
    factors_named,
    generate_all,
    held_of,
    javascript,
    no_bytecode,
    run_sheet,
    source,
)
from tests.web.api.test_book_examples import catalogue
from tests.web.api.test_charts import _limited
from tests.web.api.test_contract import curved

#: Best settings for the worked example as the catalogue gives it, its factors named in full: coded, as the page
#: sends them. The tests that make the example themselves, not through the shared fixtures, read the page's names.
EXAMPLE_BEST = {"Temperature": 1.0, "Pressure": -1.0, "Flow": 0.6}


def page_code(node: str, function: str, *args: object) -> str:
    """Return the code one of ``web/python.js``'s writers gives for the page's state."""
    return javascript(node, source("python.js"), [(function, list(args))])[0]


def panel(node: str, *steps: tuple[str, list]) -> str:
    """Return the code a panel shows: its step's code after the code of the steps it builds on."""
    codes = javascript(node, source("python.js"), list(steps))
    return javascript(node, source("python.js"), [("cumulative", codes)])[0]


def run(code: str, namespace: dict | None = None) -> dict:
    """Run code as a reader would, in ``namespace`` or a fresh one; return the namespace."""
    namespace = {} if namespace is None else namespace
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)  # noqa: S102 - the page's own code, the subject of this test
    return namespace


def coefficients_of(namespace: dict) -> dict[str, float]:
    return {c["term"]: c["coefficient"] for c in namespace["analysis"]["coefficients"]}


def assert_same_fit(namespace: dict, analysis: dict) -> None:
    """Assert that the code's analysis gives each of the page's coefficients, and no other."""
    shown = coefficients_of(namespace)
    assert set(shown) == {c["term"] for c in analysis["coefficients"]}
    for c in analysis["coefficients"]:
        assert shown[c["term"]] == pytest.approx(c["coefficient"], abs=1e-9), c["term"]


@pytest.mark.slow
def test_each_panel_s_code_runs_on_its_own_to_what_the_page_shows(api, node):
    demo = api.ok("api_catalogue")["demo"]
    held = held_of(api.ok("api_make_design", demo["design"]))
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulation["responses"]})
    design = ("designCode", [held["spec"]])
    results = ("resultsCode", [held, simulation["responses"], simulation])
    analyse = ("analyseCode", [analysis["formula"], True])

    names = [f["name"] for f in held["spec"]["factors"]]
    made = run(panel(node, design))["result"].design_actual[names].to_numpy(dtype=float)
    np.testing.assert_allclose(made, run_sheet(held)[names], atol=1e-9)

    step2 = run(panel(node, design, results))
    np.testing.assert_allclose(step2["X"][names], coded(run_sheet(held), held["spec"]["factors"]), atol=1e-12)
    np.testing.assert_allclose(step2["y"], simulation["responses"], atol=1e-9, err_msg="rounded as the page rounds")

    assert_same_fit(run(panel(node, design, results, analyse)), analysis)


@pytest.mark.slow
def test_at_a_level_of_the_study_s_own_the_code_gives_the_page_s_intervals(api, node):
    """The code names the level, so a reader who copies it gets the 99% intervals the page shows, not 95% ones."""
    demo = api.ok("api_catalogue")["demo"]
    held = held_of(api.ok("api_make_design", demo["design"]))
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    request = {**held, "responses": simulation["responses"]}
    usual, strict = (api.ok("api_analyze_runs", {**request, "significance_level": level}) for level in (0.05, 0.01))
    # The page's module for its numbers holds the study's level, and python.js reads it there.
    at_strict = source("python.js") + '\nimport { setSignificance } from "./numbers.js";\nsetSignificance(0.01);\n'
    steps = [("designCode", [held["spec"]]), ("resultsCode", [held, simulation["responses"], simulation])]
    codes = javascript(node, at_strict, [*steps, ("analyseCode", [strict["formula"], True])])
    code = javascript(node, source("python.js"), [("cumulative", codes)])[0]
    assert "significance_level=0.01" in code
    shown = {c["term"]: c for c in run(code)["analysis"]["coefficients"]}
    for c, at_usual in zip(strict["coefficients"], usual["coefficients"], strict=True):
        assert shown[c["term"]]["ci_low"] == pytest.approx(c["ci_low"], abs=1e-9), c["term"]
        assert shown[c["term"]]["ci_high"] == pytest.approx(c["ci_high"], abs=1e-9), c["term"]
        assert c["ci_low"] < at_usual["ci_low"], "the interval at 0.01 is the wider"


@pytest.mark.slow
def test_runs_still_to_do_are_left_out_of_the_analysis_as_the_page_leaves_them_out(api, node):
    held = held_of(api.ok("api_make_design", {"design_type": "ccd", "factors": FACTORS}))
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 2})
    responses = [None, *simulation["responses"][1:-1], None]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], False]),
    )
    namespace = run(code)
    assert namespace["y"].isna().tolist() == [True] + [False] * (len(responses) - 2) + [True]
    assert_same_fit(namespace, analysis)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(catalogue()))
def test_the_code_for_a_book_example_gives_the_book_s_analysis(api, node, build_script, name):
    """A book example lists its runs in the order of the book's table, not generate_design's: the code keeps it."""
    with no_bytecode():
        entry = next(e for e in build_script.book_examples() if e["name"] == name)
        campaign = build_script.book_example(entry)["campaign"]
    held = {key: campaign["design"][key] for key in ("spec", "columns", "rows")}
    responses = campaign["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses, "model": campaign["model"]})
    complete = all(v is not None for v in responses)
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], complete]),
    )
    assert_same_fit(run(code), analysis)


@pytest.mark.slow
def test_the_design_code_reproduces_every_design_the_page_offers(api, node):
    """Each design type's options, its seed included, reach the code, so the copied code gives the same runs."""
    designs = [api.ok("api_make_design", {"design_type": d, "factors": FACTORS}) for d in ALL_DESIGNS]
    codes = javascript(node, source("python.js"), [("designCode", [d["spec"]]) for d in designs])
    names = [f["name"] for f in FACTORS]
    for design_type, design, code in zip(ALL_DESIGNS, designs, codes, strict=True):
        result = run(code)["result"]
        np.testing.assert_allclose(
            result.design_actual[names].to_numpy(dtype=float), run_sheet(design)[names], err_msg=design_type
        )


@pytest.mark.parametrize("units", ['in. "wide"', "C:\\temp", "°C", "mol/L · s", ""])
def test_the_design_code_keeps_the_units_a_reader_types(api, node, units):
    """Units are free text: whatever the reader types, the code must still run and say the same."""
    spec = {"design_type": "full_factorial", "factors": [{**f, "units": units} for f in FACTORS[:2]]}
    design = api.ok("api_make_design", spec)
    factors = run(page_code(node, "designCode", design["spec"]))["factors"]
    assert [f.units for f in factors] == [units, units]


def test_results_typed_in_are_listed_in_run_order_with_none_for_runs_to_do(api, node):
    held = held_of(api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS[:2]}))
    typed = [61.25, None, -3, 1e-7] + [1.0] * (len(held["rows"]) - 4)
    y = run(page_code(node, "resultsCode", held, typed))["y"]
    assert y.name == held["spec"]["response"]
    np.testing.assert_array_equal(y.to_numpy(dtype=float)[:4], [61.25, np.nan, -3, 1e-7])


@pytest.mark.slow
@pytest.mark.parametrize(
    "spec",
    [
        {"design_type": "full_factorial", "factors": FACTORS[:2], "n_center_points": 2, "n_blocks": 2},
        {"design_type": "ccd", "factors": FACTORS, "alpha": 1.4, "n_replicates": 2},
        {"design_type": "box_behnken", "factors": FACTORS, "n_blocks": 2},
        {
            "design_type": "full_factorial",
            "factors": [*FACTORS[:2], {"name": "Supplier", "low": "A", "high": "B", "units": ""}],
            "n_center_points": 2,
            "n_blocks": 2,
        },
        {"design_type": "taguchi", "factors": FACTORS, "n_replicates": 2},
        {
            "design_type": "taguchi",
            "factors": [*FACTORS, {"name": "Supplier", "low": "A", "high": "B", "units": ""}],
            "n_levels": 3,
        },
    ],
    ids=[
        "blocks confounding an interaction",
        "typed axial distance, replicated",
        "blocks by exchange",
        "a categorical factor, in blocks",
        "a Taguchi L4, replicated",
        "a Taguchi L18 at three levels, with a categorical factor",
    ],
)
def test_the_python_shown_reproduces_blocks_replicates_and_a_typed_axial_distance(api, node, spec):
    """The design, its simulated results and their analysis, as the analysis panel shows the code for them."""
    held = held_of(api.ok("api_make_design", spec))
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 3})
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulation["responses"]})
    design = ("designCode", [held["spec"]])
    code = panel(
        node,
        design,
        ("resultsCode", [held, simulation["responses"], simulation]),
        ("analyseCode", [analysis["formula"], True]),
    )
    assert_same_fit(run(code), analysis)
    names = [f["name"] for f in held["spec"]["factors"]]
    made = run(panel(node, design))["result"].design_actual[names]
    assert made.to_numpy().tolist() == run_sheet(held)[names].to_numpy().tolist(), "the page's runs, in its order"


@pytest.mark.slow
def test_the_python_shown_reproduces_a_design_with_a_factor_that_is_hard_to_change(api, node):
    """The split-plot design, and each coefficient judged against the spread it meets, as the page judges it."""
    plan = {"design_type": "d_optimal", "factors": FACTORS, "model_type": "interactions", "budget": 12}
    held = held_of(api.ok("api_make_design", {**plan, "hard_to_change": [FACTORS[0]["name"]]}))
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 3})
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulation["responses"], "model": "main_effects"})
    assert "split_plot" in analysis, "the results are enough for two spreads"
    design = ("designCode", [held["spec"]])
    code = panel(
        node,
        design,
        ("resultsCode", [held, simulation["responses"], simulation]),
        ("analyseCode", [analysis["formula"], True, held["spec"]]),
    )
    found = run(code)["split_plot"]
    shown = {c["term"]: c for c in found["coefficients"]}
    for c in analysis["coefficients"]:
        for key in ("coefficient", "std_error", "p_value"):
            assert shown[c["term"]][key] == pytest.approx(c[key], abs=1e-9), (c["term"], key)
    tested = {t["source"]: t for t in found["tests"]}
    for t in analysis["split_plot"]["tests"]:  # the page's table of tests, term by term
        assert (tested[t["source"]]["F"], tested[t["source"]]["p_value"]) == pytest.approx((t["F"], t["p_value"]))
    names = [f["name"] for f in held["spec"]["factors"]]
    made = run(panel(node, design))["result"].design_actual
    assert made[names].to_numpy().tolist() == run_sheet(held)[names].to_numpy().tolist(), (
        "the page's runs, in its order"
    )
    assert made["WholePlot"].tolist() == run_sheet(held)["Whole plot"].tolist()


@pytest.mark.slow
def test_the_comparison_code_reproduces_the_page_s_table(api, node):
    made, held_designs = generate_all(api, FACTORS, "main_quadratic")
    compared = api.ok("api_compare", {"model": "main_quadratic", "designs": held_designs})
    designs = [
        {"label": m["label"], "design_type": m["design_type"], "options": m["design"]["spec"]["options"]} for m in made
    ]
    namespace = run(page_code(node, "compareCode", FACTORS, designs, "main_quadratic"))
    assert list(namespace["comparison"].designs) == compared["designs"]
    for label in compared["designs"]:
        assert namespace["comparison"].table[label].tolist() == compared["display"][label], label


@pytest.mark.slow
@pytest.mark.parametrize("goal", ["target", "maximize", "minimize", "learn"])
def test_the_spread_code_reproduces_the_page_s_settings_and_response_table(api, node, goal):
    """Schmidt and Launsby's 2^(3-1) read twice a setting, a reading still to do: the code sums up as the page does."""
    settings = [[1, 1, 2], [2, 1, 1], [1, 2, 1], [2, 2, 2]]
    factors = [{"name": n, "low": 1, "high": 2, "units": ""} for n in "ABC"]
    spec = {
        "design_type": "given",
        "factors": factors,
        "runs": [s for s in settings for _ in range(2)],
        "response": "y",
    }
    held = held_of(api.ok("api_make_design", {**spec, "response_info": {"goal": goal, "target": 20}}))
    responses = [20.0, 30.0, 9.0, 11.0, 26.0, 24.0, 15.0, None]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    spread = analysis["spread"]
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], False, held["spec"], None, {"spread": True}]),
    )
    namespace = run(code)
    shown = namespace["settings"].reset_index()
    assert len(shown) == len(spread["settings"]) == 3, "the setting read once is left out"
    for s in spread["settings"]:
        row = shown[(shown[["A", "B", "C"]] == [s["settings"][n] for n in "ABC"]).all(axis=1)].iloc[0]
        for key in ("n", "mean", "std_dev", "ln_std_dev", "sn"):
            if key in row:
                assert row[key] == pytest.approx(s[key]), (key, s["settings"])
    assert ("sn" in shown) == (goal != "learn")
    last = spread["factors"][-1]
    table = namespace["table"]  # the last factor's, C's
    for level in last["levels"]:
        for key in table.columns:
            assert table.loc[level["level"], key] == pytest.approx(level[key]), (key, level["level"])


#: Limits on the worked example's settings: a straight one, and a curved one.
LIMITS = [{"expression": "T + 25 * P <= 250", "note": "the product chars"}, {"expression": "T * F <= 3600"}]


@pytest.mark.slow
@pytest.mark.parametrize("design_type", ["d_optimal", "i_optimal"])
def test_the_design_code_keeps_to_the_limits_as_the_page_does(api, node, design_type):
    """The limits reach the code, so the copied code gives the page's runs, each one inside the limits."""
    design = api.ok("api_make_design", {"design_type": design_type, "factors": FACTORS, "constraints": LIMITS})
    code = page_code(node, "designCode", design["spec"])
    assert "the product chars" not in code, "a limit's note is the reader's, and stays on the page"
    runs = run(code)["result"].design_actual
    names = [f["name"] for f in FACTORS]
    np.testing.assert_allclose(runs[names].to_numpy(dtype=float), run_sheet(design)[names])
    assert ((runs["T"] + 25 * runs["P"] <= 250 + 1e-9) & (runs["T"] * runs["F"] <= 3600 + 1e-9)).all()


@pytest.mark.slow
def test_the_comparison_code_compares_over_the_settings_the_limits_leave(api, node):
    made, held_designs = generate_all(api, FACTORS, "main_quadratic", LIMITS[:1])
    compared = api.ok("api_compare", {"model": "main_quadratic", "designs": held_designs})
    designs = [
        {"label": m["label"], "design_type": m["design_type"], "options": m["design"]["spec"]["options"]} for m in made
    ]
    stored = made[0]["design"]["spec"]["constraints"]
    namespace = run(page_code(node, "compareCode", FACTORS, designs, "main_quadratic", stored))
    assert list(namespace["comparison"].designs) == compared["designs"]
    assert [m["design_type"] for m in made] == ["d_optimal", "i_optimal"], "the designs that keep every run inside"
    for label in compared["designs"]:
        assert namespace["comparison"].table[label].tolist() == compared["display"][label], label


#: Each mixture design, in three components with blank bounds, or with bounds for extreme vertices.
MIXTURES = [
    {"design_type": "simplex_lattice", "lattice_degree": 1},
    {"design_type": "simplex_lattice", "lattice_degree": 2, "n_replicates": 2},
    {"design_type": "simplex_centroid"},
    {"design_type": "extreme_vertices", "mixture_model": "scheffe_linear"},
    {"design_type": "extreme_vertices", "mixture_model": "scheffe_quadratic"},
]


def components(design_type: str) -> list[dict]:
    """Return three components, each from 0 to 1, or with the bounds extreme vertices need."""
    if design_type == "extreme_vertices":
        bounds = [(0.2, 0.7), (0.1, 0.6), (0.0, 0.5)]
        return [{"name": n, "low": lo, "high": hi, "units": "%"} for n, (lo, hi) in zip("ABC", bounds, strict=True)]
    return [{"name": n, "low": None, "high": None, "units": ""} for n in ("Water", "Oil", "Wax")]


@pytest.mark.slow
@pytest.mark.parametrize("request_", MIXTURES, ids=lambda r: "-".join(str(v) for v in r.values()))
def test_the_design_code_reproduces_each_mixture_the_page_makes(api, node, request_):
    """A mixture's code makes its components, each a proportion, and the library's mixture design for its model."""
    factors = components(request_["design_type"])
    design = api.ok("api_make_design", {**request_, "factors": factors})
    code = page_code(node, "designCode", design["spec"])
    assert 'design_type="mixture"' in code
    names = [f["name"] for f in factors]
    namespace = run(code)
    assert [f.type for f in namespace["factors"]] == ["mixture"] * 3
    made = namespace["result"].design_actual[names].to_numpy(dtype=float)
    np.testing.assert_allclose(made, run_sheet(design)[names], atol=1e-11)


@pytest.mark.slow
@pytest.mark.parametrize("request_", MIXTURES[1:3], ids=["lattice", "centroid"])
def test_the_python_shown_reproduces_a_mixture_s_simulation_and_its_scheffe_fit(api, node, request_):
    """The proportions are fitted as they are, not coded, with the Scheffé model the page fitted."""
    held = held_of(api.ok("api_make_design", {**request_, "factors": components(request_["design_type"])}))
    simulation = api.ok("api_simulate", {**held, "seed": 5})
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulation["responses"]})
    assert analysis["formula"] == "scheffe_quadratic"
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, simulation["responses"], simulation]),
        ("analyseCode", [analysis["formula"], True]),
    )
    namespace = run(code)
    names = [f["name"] for f in held["spec"]["factors"]]
    np.testing.assert_allclose(namespace["X"][names], run_sheet(held)[names].astype(float), atol=1e-12)
    np.testing.assert_allclose(namespace["y"], simulation["responses"], atol=1e-9)
    assert_same_fit(namespace, analysis)
    assert "Intercept" not in coefficients_of(namespace)


@pytest.mark.slow
def test_the_python_shown_reproduces_each_response_of_a_mixture(api, node):
    """Each response of a mixture is simulated from its own process, and its code fits its own column."""
    responses = [{"name": "Hardness", "goal": "maximize"}, {"name": "Cost", "goal": "minimize"}]
    request = {**MIXTURES[1], "factors": components("simplex_lattice"), "responses": responses}
    held = held_of(api.ok("api_make_design", request))
    simulation = api.ok("api_simulate", {**held, "seed": 5})
    results = {"Hardness": simulation["responses"], "Cost": simulation["more"]["Cost"]["responses"]}
    sent = {"responses": results["Hardness"], "more_responses": {"Cost": results["Cost"]}}
    analysis = api.ok("api_analyze_runs", {**held, **sent, "response": "Cost"})
    spec = javascript(node, source("python.js"), [("forResponse", [held["spec"], "Cost"])])[0]
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results, simulation]),
        ("analyseCode", [analysis["formula"], True, spec]),
    )
    namespace = run(code)
    for name, values in results.items():
        np.testing.assert_allclose(namespace["runs"][name], values, atol=1e-9, err_msg=name)
    assert namespace["y"].name == "Cost"
    assert_same_fit(namespace, analysis)


def notebook_of(node: str, **campaign: object) -> dict:
    """Return the notebook ``web/python.js`` writes for a campaign, as the page downloads it."""
    defaults = {"title": "A design", "version": "1.2.3", "url": "https://example.org/app/", "simulation": None}
    return javascript(node, source("python.js"), [("notebook", [{**defaults, **campaign}])])[0]


@pytest.mark.slow
def test_the_notebook_runs_top_to_bottom_to_the_page_s_analysis(api, node):
    """The notebook a reader downloads is each panel's code in order, after the line that installs the release."""
    demo = api.ok("api_catalogue")["demo"]
    design = api.ok("api_make_design", demo["design"])
    held = held_of(design)
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulation["responses"]})
    sources = ["simulated"] * len(simulation["responses"])
    book = notebook_of(
        node,
        design=held,
        responses=simulation["responses"],
        sources=sources,
        simulation=simulation,
        model=analysis["formula"],
    )
    nbformat = pytest.importorskip("nbformat", reason="the notebook is checked against Jupyter's own schema")
    nbformat.validate(nbformat.reads(json.dumps(book), as_version=4))
    code = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
    assert code[0] == '%pip install "process-improve==1.2.3"', "the release the page ran"
    assert code[1:] == [
        page_code(node, "designCode", design["spec"]),
        page_code(node, "resultsCode", held, simulation["responses"], simulation),
        page_code(node, "analyseCode", analysis["formula"], True),
    ]
    namespace: dict = {}
    for cell in code[1:]:
        run(cell, namespace)
    assert_same_fit(namespace, analysis)


def test_a_notebook_of_typed_results_writes_them_out_and_waits_for_an_analysis(api, node):
    """Results typed in, or a simulation edited, are written out; with no analysis yet there is no analysis cell."""
    held = held_of(
        api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 0})
    )
    responses = [1.5, None, 2.5, 3.0, None, 4.0, 4.5, 5.0]
    book = notebook_of(node, design=held, responses=responses, sources=["entered"] * 8, model=None)
    code = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
    assert code[-1] == page_code(node, "resultsCode", held, responses)
    assert not any("analyze_experiment" in cell for cell in code)
    unpinned = notebook_of(node, design=held, responses=responses, sources=["entered"] * 8, model=None, version=None)
    assert "".join(unpinned["cells"][1]["source"]) == "%pip install process-improve"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("spec", "blank"),
    [
        (correlated_study(), ()),
        ({"design_type": "full_factorial", "factors": FACTORS, "n_blocks": 2, "n_center_points": 2}, (0, 3)),
    ],
    ids=["correlated runs as given", "blocks, with runs still to do"],
)
def test_the_code_gives_the_vifs_the_page_shows(api, node, spec, blank):
    """When the coefficients table shows each term's VIF, the analysis panel's code works the same VIFs out."""
    held = held_of(api.ok("api_make_design", spec))
    responses = [None if i in blank else float(i % 3) + 0.5 * i for i in range(len(held["rows"]))]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses, "model": "main_effects"})
    complete = not blank
    (vif_model,) = javascript(node, source("logic.js"), [("fittedFormula", [analysis["coefficients"]])])
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], complete, None, vif_model]),
    )
    namespace = run(code)
    assert_same_fit(namespace, analysis)
    found = namespace["vif"]
    blocks = [v for key, v in found.items() if key.startswith("Block[")]  # the library's names for the contrasts
    terms = {key: v for key, v in found.items() if not key.startswith("Block[")}
    shown = [c for c in analysis["coefficients"] if c["term"] != "Intercept"]
    assert {c["term"]: c["vif"] for c in shown if not c["block"]} == {k: pytest.approx(v) for k, v in terms.items()}
    assert [c["vif"] for c in shown if c["block"]] == pytest.approx(blocks), "in the order of the blocks"
    assert len(blocks) == spec.get("n_blocks", 1) - 1


def test_the_code_for_runs_as_given_lists_the_runs_and_gives_the_page_s_analysis(api, node):
    """Runs as given are made by no call: step 1 holds the factors and the limits, step 2 the runs themselves."""
    runs = [[t, p, f] for t in (150, 200) for p in (1, 3) for f in (10, 20)] + [[175, 2, 15]] * 3
    spec = {
        "design_type": "given",
        "factors": FACTORS,
        "runs": runs,
        "constraints": [{"expression": "T + 25*P <= 300"}],
    }
    held = held_of(api.ok("api_make_design", spec))
    responses = [float(i % 5) + 0.1 * i for i in range(len(runs))]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    step_one = run(page_code(node, "designCode", held["spec"]))
    assert [f.name for f in step_one["factors"]] == ["T", "P", "F"]
    assert step_one["constraints"] == [held["spec"]["constraints"][0]["expression"]], "as Python wrote it"
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], True]),
    )
    assert_same_fit(run(code), analysis)


# --------------------------------------------------------------------------- the best settings for the goal
def aimed(held: dict, goal: str, target: float | None = None) -> dict:
    """Give the run sheet's response a goal, as the brief does."""
    info = {"label": "Yield", "units": "%", "goal": goal, **({"target": target} if target is not None else {})}
    return {**held, "spec": {**held["spec"], "response_info": info}}


def assert_same_best(namespace: dict, best: dict, spec: dict) -> None:
    """Assert that the code's optimiser finds the page's best settings, and the page's prediction and intervals."""
    found = namespace["best"]
    for setting, factor in zip(best["settings"], spec["factors"], strict=True):
        if setting["kind"] == "numeric":
            tolerance = 1e-6 * (factor["high"] - factor["low"])
            assert found["optimal_actual"][setting["name"]] == pytest.approx(setting["value"], abs=tolerance)
        elif setting["kind"] == "component":  # a mixture's proportions, which its Scheffé model is fitted on
            assert found["optimal_coded"][setting["name"]] == pytest.approx(setting["value"], abs=1e-6)
    assert found["predicted_responses"][spec["response"]] == pytest.approx(best["predicted"], abs=1e-6)
    shown = namespace["prediction"]["predictions"][0]
    assert shown["predicted"] == pytest.approx(best["predicted"], abs=1e-6)
    if best["confidence"] is not None:
        assert [shown["ci_low"], shown["ci_high"]] == pytest.approx(best["confidence"], abs=1e-6)
        assert [shown["pi_low"], shown["pi_high"]] == pytest.approx(best["prediction"], abs=1e-6)


def best_panel(node: str, held: dict, responses: list, replies: tuple[dict, dict], simulation=None) -> str:
    """Return the code the best settings' panel shows: every step, the best settings last.

    ``replies`` are the page's analysis and its best settings, from the same results.
    """
    analysis, best = replies
    complete = all(v is not None for v in responses)
    return panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses, simulation]),
        ("analyseCode", [analysis["formula"], complete]),
        ("bestCode", [best, held["spec"], analysis["formula"], complete]),
    )


def test_at_a_level_of_the_study_s_own_the_best_settings_code_gives_the_page_s_intervals(api, node, worked_example):
    """At 0.01 the page shows 99% intervals at the best settings; the code it shows names the level and gives them."""
    held = aimed(worked_example["held"], "maximize")
    simulation = worked_example["simulated"]
    request = {**held, "responses": simulation["responses"], "significance_level": 0.01}
    analysis, best = api.ok("api_analyze_runs", request), api.ok("api_best_settings", request)
    assert best["level"] == pytest.approx(0.99)
    at_strict = source("python.js") + '\nimport { setSignificance } from "./numbers.js";\nsetSignificance(0.01);\n'
    steps = [
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, simulation["responses"], simulation]),
        ("analyseCode", [analysis["formula"], True]),
        ("bestCode", [best, held["spec"], analysis["formula"], True]),
    ]
    code = javascript(node, source("python.js"), [("cumulative", javascript(node, at_strict, steps))])[0]
    namespace = run(code)
    assert_same_fit(namespace, analysis)
    assert_same_best(namespace, best, held["spec"])


@pytest.mark.parametrize(("goal", "target"), [("maximize", None), ("minimize", None), ("target", 60.0)])
def test_the_best_settings_code_finds_the_page_s_best_settings(api, node, worked_example, goal, target):
    held = aimed(worked_example["held"], goal, target)
    simulation = worked_example["simulated"]
    request = {**held, "responses": simulation["responses"]}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)
    namespace = run(best_panel(node, held, simulation["responses"], (analysis, best), simulation))
    assert_same_fit(namespace, analysis)
    assert_same_best(namespace, best, held["spec"])


@pytest.mark.slow
@pytest.mark.parametrize("transform", ["log", "sqrt", "inverse"])
def test_a_transformed_response_s_code_fits_and_finds_what_the_page_does(api, node, worked_example, transform):
    """The code fits the response's transform, as the page does, and its prediction put back is the page's."""
    held = aimed(worked_example["held"], "maximize")
    held = {**held, "spec": {**held["spec"], "analysis_transforms": {held["spec"]["response"]: transform}}}
    simulation = worked_example["simulated"]
    request = {**held, "responses": simulation["responses"]}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, simulation["responses"], simulation]),
        ("analyseCode", [analysis["formula"], True, held["spec"]]),  # as app.js passes the response's spec
        ("bestCode", [best, held["spec"], analysis["formula"], True]),
    )
    namespace = run(code)
    assert_same_fit(namespace, analysis)
    back = {"log": np.exp, "sqrt": np.square, "inverse": lambda v: -1 / v}[transform]
    shown = namespace["prediction"]["predictions"][0]
    assert back(shown["predicted"]) == pytest.approx(best["predicted"], rel=1e-6)
    for setting in best["settings"]:
        if setting["kind"] == "numeric":
            assert namespace["best"]["optimal_actual"][setting["name"]] == pytest.approx(setting["value"], rel=1e-6)


def test_the_notebook_ends_with_the_best_settings_when_the_page_has_them(api, node, worked_example):
    held = aimed(worked_example["held"], "maximize")
    simulation = worked_example["simulated"]
    request = {**held, "responses": simulation["responses"]}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)
    common = {"design": held, "responses": simulation["responses"], "sources": ["simulated"] * 15}
    book = notebook_of(node, **common, simulation=simulation, model=analysis["formula"], best=best)
    titles = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "markdown"]
    assert titles[-1] == "## 4. The best settings for the goal"
    namespace: dict = {}
    for cell in book["cells"][2:]:
        if cell["cell_type"] == "code":
            run("".join(cell["source"]), namespace)
    assert_same_best(namespace, best, held["spec"])
    learn = api.ok(
        "api_best_settings", {**aimed(worked_example["held"], "learn"), "responses": simulation["responses"]}
    )
    plain = notebook_of(node, **common, simulation=simulation, model=analysis["formula"], best=learn)
    assert not any("optimize_responses" in "".join(cell["source"]) for cell in plain["cells"]), "no goal, no step 4"


def test_the_best_settings_code_holds_a_categorical_factor_and_the_blocks_as_the_page_does(api, node):
    factors = [*FACTORS[:2], {"name": "Supplier", "low": "A", "high": "B", "units": ""}]
    spec = {"design_type": "full_factorial", "factors": factors, "n_center_points": 2, "n_blocks": 2}
    held = aimed(held_of(api.ok("api_make_design", spec)), "maximize")
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 3})
    request = {**held, "responses": simulation["responses"]}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)
    assert set(best["search"]["held"]) == {"Block1", "Supplier"}, "the case holds both"
    code = best_panel(node, held, simulation["responses"], (analysis, best), simulation)
    assert "def held_at(coefficients, held):" in code
    assert_same_best(run(code), best, held["spec"])


def test_the_best_settings_code_keeps_to_the_limits_on_the_settings(api, node):
    spec = {"design_type": "d_optimal", "factors": FACTORS, "constraints": LIMITS[:1]}
    held = aimed(held_of(api.ok("api_make_design", spec)), "maximize")
    sheet = run_sheet(held)
    responses = (50 + 3 * (sheet["T"] - 175) / 25 + 2 * (sheet["P"] - 2) - ((sheet["F"] - 15) / 5) ** 2).tolist()
    request = {**held, "responses": responses}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)
    assert best["limits"], "the case is best on its limit"
    code = best_panel(node, held, responses, (analysis, best))
    assert "region=DesignRegion(factors, constraints)" in code
    assert_same_best(run(code), best, held["spec"])


def test_the_best_settings_code_leaves_the_confirmation_runs_out_of_the_fit(api, node, worked_example):
    held = aimed(worked_example["held"], "maximize")
    simulation = worked_example["simulated"]
    best = api.ok("api_best_settings", {**held, "responses": simulation["responses"]})
    at = best["confirm_at"]
    columns = [*held["columns"], "Confirmation run"]
    extra = [[{"Run order": 16 + i, "Confirmation run": "yes", **at}.get(c) for c in columns] for i in range(3)]
    sheet = {**held, "columns": columns, "rows": [*[[*row, None] for row in held["rows"]], *extra]}
    responses = [*simulation["responses"], 60.0, 61.0, None]
    analysis = api.ok("api_analyze_runs", {**sheet, "responses": responses})
    again = api.ok("api_best_settings", {**sheet, "responses": responses})
    namespace = run(best_panel(node, sheet, responses, (analysis, again)))
    assert len(namespace["y"]) == 15, "the design's runs only"
    assert_same_fit(namespace, analysis)
    assert_same_best(namespace, again, held["spec"])


@pytest.mark.slow
@pytest.mark.parametrize(
    ("request_", "goal", "target"),
    [(MIXTURES[1], "maximize", None), (MIXTURES[2], "target", 50.0), (MIXTURES[4], "minimize", None)],
    ids=["lattice", "centroid-target", "vertices"],
)
def test_the_best_blend_code_finds_the_page_s_best_blend(api, node, request_, goal, target):
    """A mixture's code searches the region of blends: each component within its bounds, the proportions sum to 1."""
    made = held_of(api.ok("api_make_design", {**request_, "factors": components(request_["design_type"])}))
    simulation = api.ok("api_simulate", {**made, "seed": 5})
    held = aimed(made, goal, target)
    request = {**held, "responses": simulation["responses"]}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)
    assert {s["kind"] for s in best["settings"]} == {"component"}
    code = best_panel(node, held, simulation["responses"], (analysis, best), simulation)
    assert "region=DesignRegion(factors)" in code
    namespace = run(code)
    assert_same_fit(namespace, analysis)
    assert_same_best(namespace, best, held["spec"])
    assert sum(namespace["best"]["optimal_coded"].values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- several responses
#: Three responses, as the brief lists them: a yield to maximise, a cost to minimise and a purity with a target.
THREE = [
    {"name": "Yield", "label": "Yield", "units": "%", "goal": "maximize"},
    {"name": "Cost", "label": "Cost", "units": "$/kg", "goal": "minimize"},
    {"name": "Purity", "label": "Purity", "units": "%", "goal": "target", "target": 63.0},
]


def three_responses(api) -> tuple[dict, dict]:
    """Make a Box-Behnken design of the three responses, and simulate their results; return it and the simulation."""
    held = held_of(api.ok("api_make_design", {"design_type": "box_behnken", "factors": FACTORS, "responses": THREE}))
    return held, api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})


def by_name(simulation: dict) -> dict[str, list]:
    """Return each response's results by its column, as the page holds them."""
    return {"Yield": simulation["responses"], **{n: m["responses"] for n, m in simulation["more"].items()}}


def sent_of(results: dict[str, list]) -> dict:
    """Return the results as the page sends them: the first response's alone, the later ones' beside them."""
    return {"responses": results["Yield"], "more_responses": {n: results[n] for n in ("Cost", "Purity")}}


@pytest.mark.slow
def test_the_code_simulates_each_response_from_its_own_process_as_the_page_does(api, node):
    held, simulation = three_responses(api)
    code = panel(node, ("designCode", [held["spec"]]), ("resultsCode", [held, by_name(simulation), simulation]))
    namespace = run(code)
    for name, values in by_name(simulation).items():
        np.testing.assert_allclose(namespace["runs"][name], values, atol=1e-9, err_msg=name)
    np.testing.assert_allclose(namespace["y"], simulation["responses"], atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("response", ["Cost", "Purity"])
def test_each_response_s_analysis_code_gives_the_page_s_analysis_of_it(api, node, response):
    """A later response's results are typed in, one still to come: its code takes its own column and leaves it out."""
    held, simulation = three_responses(api)
    results = by_name(simulation)
    results[response] = [None, *results[response][1:]]
    analysis = api.ok("api_analyze_runs", {**held, **sent_of(results), "response": response, "model": "interactions"})
    spec = javascript(node, source("python.js"), [("forResponse", [held["spec"], response])])[0]
    assert spec["response"] == response
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results]),
        ("analyseCode", [analysis["formula"], False, spec]),
    )
    namespace = run(code)
    assert namespace["y"].name == response
    assert_same_fit(namespace, analysis)


@pytest.mark.slow
def test_each_response_s_best_settings_code_finds_the_page_s(api, node):
    held, simulation = three_responses(api)
    results = by_name(simulation)
    request = {**held, **sent_of(results), "response": "Cost", "model": "interactions"}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)
    spec = javascript(node, source("python.js"), [("forResponse", [held["spec"], "Cost"])])[0]
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results, simulation]),
        ("analyseCode", [analysis["formula"], True, spec]),
        ("bestCode", [best, spec, analysis["formula"], True]),
    )
    namespace = run(code)
    found = namespace["best"]["optimal_coded"]
    for name in ("T", "P", "F"):
        assert found[name] == pytest.approx(best["coded"][name], abs=1e-6), name


@pytest.mark.slow
def test_the_notebook_of_three_responses_analyses_each_in_turn(api, node):
    held, simulation = three_responses(api)
    results = by_name(simulation)
    analyses, fits = [], {}
    for name in ("Yield", "Cost", "Purity"):
        request = {**held, **sent_of(results), "response": name}
        fits[name] = api.ok("api_analyze_runs", request)
        best = api.ok("api_best_settings", request)
        analyses.append({"response": name, "model": fits[name]["formula"], "best": best})
    # Cost's table shows its VIFs, as the page shows them when a term is correlated: its cell works them out.
    (terms,) = javascript(node, source("logic.js"), [("fittedFormula", [fits["Cost"]["coefficients"]])])
    analyses[1]["vifModel"] = terms
    marks = {name: ["simulated"] * 15 for name in results}
    book = notebook_of(
        node, design=held, responses=results, sources=marks, simulation=simulation, model=None, analyses=analyses
    )
    nbformat = pytest.importorskip("nbformat", reason="the notebook is checked against Jupyter's own schema")
    nbformat.validate(nbformat.reads(json.dumps(book), as_version=4))
    headings = ["".join(c["source"]) for c in book["cells"] if c["cell_type"] == "markdown"][3:]
    assert headings == [
        "## 3. The analysis of Yield",
        "## 4. The best settings for Yield",
        "## 5. The analysis of Cost",
        "## 6. The best settings for Cost",
        "## 7. The analysis of Purity",
        "## 8. The best settings for Purity",
    ]
    namespace: dict = {}
    for cell in book["cells"][2:]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        run(text, namespace)
        if "analysis = analyze_experiment" in text:
            assert_same_fit(namespace, fits[str(namespace["y"].name)])
            assert ("vif = evaluate_design" in text) == (namespace["y"].name == "Cost"), "the VIFs in Cost's cell only"
    vifs = {c["term"]: c["vif"] for c in fits["Cost"]["coefficients"] if c.get("vif") is not None}
    assert vifs, "the analysis gives each term's VIF"
    for term, vif in vifs.items():
        assert namespace["vif"][term] == pytest.approx(vif), term


def assert_same_balance(namespace: dict, balance: dict, numeric: list[str]) -> None:
    """Assert that the code's balance finds the page's settings, each response's d, and D."""
    found = namespace["balanced"]
    for name in numeric:
        assert found["optimal_coded"][name] == pytest.approx(balance["coded"][name], abs=1e-6), name
    assert found["composite_desirability"] == pytest.approx(balance["composite"], abs=1e-9)
    for r in balance["responses"]:
        assert found["individual_desirability"][r["name"]] == pytest.approx(r["d"], abs=1e-9), r["name"]


@pytest.mark.slow
def test_the_balance_code_finds_the_page_s_balance(api, node):
    held, simulation = three_responses(api)
    sent = sent_of(by_name(simulation))
    goals = {"Yield": {"importance": 5, "shape": "more"}, "Cost": {"high": 70.0}}
    request = {**held, **sent, "models": {"Cost": "interactions"}, "goals": goals}
    balance = api.ok("api_balance", request)
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, by_name(simulation), simulation]),
        ("balanceCode", [balance, held["spec"]]),
    )
    assert_same_balance(run(code), balance, ["T", "P", "F"])


@pytest.mark.slow
def test_the_balance_code_holds_a_categorical_factor_and_keeps_to_the_limits(api, node):
    factors = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]
    spec = {"design_type": "full_factorial", "factors": factors, "responses": THREE[:2], "n_center_points": 3}
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), factors)
    yields = (60 + 4 * x["T"] - 2 * x["P"] + 3 * x["S"] + x["T"] * x["P"]).tolist()
    costs = (20 + 2 * x["T"] + x["P"] + 0.5 * x["S"]).tolist()
    request = {**held, "responses": yields, "more_responses": {"Cost": costs}, "models": {"Yield": "interactions"}}
    balance = api.ok("api_balance", request)
    assert balance["search"]["held"] == {"S": 1.0}
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, {"Yield": yields, "Cost": costs}]),
        ("balanceCode", [balance, held["spec"]]),
    )
    assert "held_at(" in code
    assert_same_balance(run(code), balance, ["T", "P"])

    limit = [{"expression": "T + 25*P <= 215"}]
    held = held_of(
        api.ok(
            "api_make_design",
            {"design_type": "d_optimal", "factors": FACTORS[:2], "responses": THREE[:2], "constraints": limit},
        )
    )
    x = coded(run_sheet(held), FACTORS[:2])
    yields = (60 + 4 * x["T"] + 2 * x["P"] - 3 * x["T"] ** 2).tolist()
    costs = (20 + 3 * x["T"] + x["P"]).tolist()
    balance = api.ok("api_balance", {**held, "responses": yields, "more_responses": {"Cost": costs}})
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, {"Yield": yields, "Cost": costs}]),
        ("balanceCode", [balance, held["spec"]]),
    )
    assert "region=DesignRegion(factors, constraints)" in code
    assert_same_balance(run(code), balance, ["T", "P"])


@pytest.mark.slow
def test_the_balance_code_fits_a_factor_nested_in_another_as_the_page_does(api, node):
    """S + P + S:T for Yield, T's effect within each level of S: the code adds S's indicators to X, as the page does."""
    levels = {"name": "S", "type": "categorical", "levels": ["a", "b", "c"], "units": ""}
    spec = {
        "design_type": "full_factorial",
        "factors": [*FACTORS[:2], levels],
        "responses": THREE[:2],
        "n_center_points": 0,
        "n_replicates": 2,
    }
    held = held_of(api.ok("api_make_design", spec))
    sheet = run_sheet(held)
    x = coded(sheet, FACTORS[:2])
    slope = sheet["S"].map({"a": 1.0, "b": 3.0, "c": 5.0})
    yields = (60 + slope * x["T"] + 2 * x["P"] + np.arange(len(sheet)) % 4 * 0.1).tolist()
    costs = (20 + 2 * x["T"] + x["P"] + np.arange(len(sheet)) % 3 * 0.1).tolist()
    models = {"Yield": {"terms": [["S"], ["P"], ["S", "T"]]}, "Cost": "main_effects"}
    request = {**held, "responses": yields, "more_responses": {"Cost": costs}, "models": models}
    balance = api.ok("api_balance", request)
    tried = api.ok("api_trade_offs", {**request, "at": balance["coded"]})
    view = {"asked": tried["asked"], "across": "Yield", "up": "Cost", "colour": None, "limits": {}}
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, {"Yield": yields, "Cost": costs}]),
        ("balanceCode", [balance, held["spec"]]),
    )
    assert 'X["S_I1"] = (runs["S"] == "a").astype(float)' in code
    assert_same_balance(run(code), balance, ["T", "P"])
    # The trade-off chart's code follows the balance's, and now runs on its fits too.
    steps = [("designCode", [held["spec"]]), ("resultsCode", [held, {"Yield": yields, "Cost": costs}])]
    charted = panel(
        node, *steps, ("balanceCode", [balance, held["spec"]]), ("tradeOffCode", [view, balance, held["spec"]])
    )
    with contextlib.redirect_stdout(io.StringIO()):
        cloud = run(charted.replace("figure.show()", ""))["cloud"]
    for name, shown in tried["candidates"]["responses"].items():
        np.testing.assert_allclose(cloud[name], shown, rtol=1e-5)


@pytest.mark.slow
def test_the_trade_off_chart_s_code_draws_the_chart_at_another_level_as_the_page_does(api, node):
    """The chart at supplier A while the balance is at B: the code holds each fitted model at A, and draws no star."""
    factors = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]
    spec = {"design_type": "full_factorial", "factors": factors, "responses": THREE[:2], "n_center_points": 3}
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), factors)
    yields = (60 + 4 * x["T"] - 2 * x["P"] + 3 * x["S"] + x["T"] * x["P"]).tolist()
    costs = (20 + 2 * x["T"] + x["P"] + 0.5 * x["S"]).tolist()
    request = {**held, "responses": yields, "more_responses": {"Cost": costs}, "models": {"Yield": "interactions"}}
    balance = api.ok("api_balance", request)
    tried = api.ok("api_trade_offs", {**request, "at": {**balance["coded"], "S": -1.0}})
    assert tried["held"] == [{"name": "S", "value": "A"}]
    view = {"asked": tried["asked"], "across": "Yield", "up": "Cost", "colour": None, "limits": {}}
    steps = [("designCode", [held["spec"]]), ("resultsCode", [held, {"Yield": yields, "Cost": costs}])]
    view_at = {**view, "held": tried["held_columns"]}
    code = panel(
        node, *steps, ("balanceCode", [balance, held["spec"]]), ("tradeOffCode", [view_at, balance, held["spec"]])
    )
    assert "star[" not in code.split("cloud, star")[1].split("figure = go.Figure(")[0], "no star at another level"
    with contextlib.redirect_stdout(io.StringIO()):
        namespace = run(code.replace("figure.show()", ""))
    for name, shown in tried["candidates"]["responses"].items():
        np.testing.assert_allclose(namespace["cloud"][name], shown, rtol=1e-5)
    names = [trace.name for trace in namespace["figure"].data]
    assert "Balanced settings" not in names, "no star at another level"
    np.testing.assert_array_equal(namespace["far"], np.asarray(tried["candidates"]["far"], dtype=bool))


@pytest.mark.slow
def test_the_balance_code_finds_the_page_s_balanced_blend(api, node):
    """A mixture's two responses are balanced over its blends, each with its own Scheffé model."""
    responses = [{"name": "Hardness", "goal": "maximize"}, {"name": "Cost", "goal": "minimize"}]
    request = {**MIXTURES[1], "factors": components("simplex_lattice"), "responses": responses}
    held = held_of(api.ok("api_make_design", request))
    simulation = api.ok("api_simulate", {**held, "seed": 5})
    results = {"Hardness": simulation["responses"], "Cost": simulation["more"]["Cost"]["responses"]}
    goals = {"Cost": {"importance": 5}}
    sent = {"responses": results["Hardness"], "more_responses": {"Cost": results["Cost"]}}
    balance = api.ok("api_balance", {**held, **sent, "models": {"Cost": "scheffe_linear"}, "goals": goals})
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results, simulation]),
        ("balanceCode", [balance, held["spec"]]),
    )
    assert "region=DesignRegion(factors)" in code
    namespace = run(code)
    assert_same_balance(namespace, balance, ["Water", "Oil", "Wax"])
    assert sum(namespace["balanced"]["optimal_coded"].values()) == pytest.approx(1.0)


@pytest.mark.slow
def test_the_notebook_ends_with_the_balance_when_the_page_has_one(api, node):
    held, simulation = three_responses(api)
    results = by_name(simulation)
    balance = api.ok("api_balance", {**held, **sent_of(results)})
    marks = {name: ["simulated"] * 15 for name in results}
    book = notebook_of(
        node, design=held, responses=results, sources=marks, simulation=simulation, model=None, balance=balance
    )
    assert "".join(book["cells"][-2]["source"]) == "## 3. The balance of the responses"
    namespace: dict = {}
    for cell in book["cells"][2:]:
        if cell["cell_type"] == "code":
            run("".join(cell["source"]), namespace)
    assert_same_balance(namespace, balance, ["T", "P", "F"])


# --------------------------------------------------------------------------- one factor, its two levels compared
SUPPLIER = {"name": "Supplier", "low": "A", "high": "B", "units": ""}


@pytest.mark.parametrize("blank", [False, True], ids=["every run", "a run still to do"])
def test_the_code_for_one_factor_tests_its_two_levels_as_the_page_does(api, node, bootstrap, blank):
    """Two suppliers, five runs each: the code's four tests give the page's intervals and p-values."""
    spec = {"design_type": "full_factorial", "factors": [SUPPLIER], "n_replicates": 5, "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    simulated = api.ok("api_simulate", {**held, "scenario": "main_effects", "sigma": 1, "seed": 2})
    responses = [None, *simulated["responses"][1:]] if blank else simulated["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], not blank, held["spec"], None, {"compared": True}]),
    )
    shown = run(code)
    assert_same_fit(shown, analysis)
    tests = {t["test"]: t for t in analysis["comparison"]["difference"]["tests"]}
    for name in ("pooled", "welch"):
        assert shown[name]["p value"] == pytest.approx(tests[name]["p_value"])
        assert [shown[name]["ConfInt: Lo"], shown[name]["ConfInt: Hi"]] == pytest.approx(
            [tests[name]["ci_low"], tests[name]["ci_high"]]
        )
    assert shown["shuffled"]["p_value_two_sided"] == pytest.approx(tests["randomization"]["p_value"])
    ranks = shown["ranks"]
    exact = 2 * min(ranks["exact_p_value_less"], ranks["exact_p_value_greater"])
    assert min(1.0, exact) == pytest.approx(tests["rank_sum"]["p_value"])
    assert javascript(node, source("python.js"), [("ARRANGEMENTS", [])]) == [bootstrap.ARRANGEMENTS]


#: Box, Hunter and Hunter's boys' shoes: a boy is a block of two runs, a sole of each material.
MATERIAL = {"name": "material", "low": "A", "high": "B", "units": ""}
SOLES = [13.2, 14.0, 8.2, 8.8, 10.9, 11.2, 14.3, 14.2, 10.7, 11.8, 6.6, 6.4, 9.5, 9.8, 10.8, 11.3, 8.8, 9.3, 13.3, 13.6]


@pytest.mark.parametrize("blank", [False, True], ids=["every run", "a run still to do"])
def test_the_code_for_two_levels_in_pairs_tests_them_within_each_pair_as_the_page_does(api, node, blank):
    """Ten boys, a sole of each material: the code's three tests give the page's interval and p-values."""
    runs = [[level] for _ in range(10) for level in ("A", "B")]
    blocks = [boy for boy in range(1, 11) for _ in (0, 1)]
    spec = {"design_type": "given", "factors": [MATERIAL], "runs": runs, "blocks": blocks}
    held = held_of(api.ok("api_make_design", spec))
    responses = [None, *SOLES[1:]] if blank else SOLES
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], not blank, held["spec"], None, {"compared": True}]),
    )
    shown = run(code)
    assert_same_fit(shown, analysis)
    tests = {t["test"]: t for t in analysis["comparison"]["difference"]["tests"]}
    assert shown["paired"]["p value"] == pytest.approx(tests["paired"]["p_value"])
    assert [shown["paired"]["ConfInt: Lo"], shown["paired"]["ConfInt: Hi"]] == pytest.approx(
        [tests["paired"]["ci_low"], tests["paired"]["ci_high"]]
    )
    assert shown["swapped"]["p_value_two_sided"] == pytest.approx(tests["sign_flip"]["p_value"])
    assert shown["ranks"]["p_value_two_sided"] == pytest.approx(tests["signed_rank"]["p_value"])


def test_a_study_of_more_factors_shows_no_comparison_code(node):
    assert "randomization_test" not in page_code(node, "analyseCode", "interactions", True)


# --------------------------------------------------------------------------- one factor of three levels or more
LOT = {"name": "Lot", "type": "categorical", "levels": ["1", "2", "3", "4"], "units": ""}


@pytest.mark.parametrize("blank", [False, True], ids=["every run", "a run still to do"])
def test_the_code_for_one_factor_of_four_levels_tests_them_as_the_page_does(api, node, blank):
    """Four lots, four runs each, their labels numbers: the code's tests give the page's, against the control picked."""
    spec = {"design_type": "full_factorial", "factors": [LOT], "n_replicates": 4, "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    simulated = api.ok("api_simulate", {**held, "scenario": "main_effects", "sigma": 1, "seed": 2})
    responses = [None, *simulated["responses"][1:]] if blank else simulated["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses, "control": "3"})
    several = analysis["comparison"]["several"]
    assert several["control"] == "3"
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], not blank, held["spec"], None, {"compared": True, "control": "3"}]),
    )
    assert '"Lot_L1": runs["Lot"].map({"1": 1.0, "2": 0.0, "3": 0.0, "4": -1.0}),' in code
    shown = run(code)
    assert_same_fit(shown, analysis)
    assert shown["anova"].statistic == pytest.approx(several["test"]["f_value"])
    assert shown["anova"].pvalue == pytest.approx(several["test"]["p_value"])
    tukey, low_high = shown["tukey"], shown["tukey"].confidence_interval(0.95)
    order = LOT["levels"]
    for pair in several["pairs"]:
        i, j = order.index(pair["from"]), order.index(pair["to"])  # the page's pair is the later level less the earlier
        assert tukey.statistic[j, i] == pytest.approx(pair["estimate"])
        assert [low_high.low[j, i], low_high.high[j, i]] == pytest.approx([pair["ci_low"], pair["ci_high"]])
        assert tukey.pvalue[j, i] == pytest.approx(pair["p_value"], abs=1e-6)
    checks = {check["test"]: check for check in several["checks"]}
    for name, check in (("spread", checks["bartlett"]), ("ranks", checks["kruskal_wallis"])):
        found = [shown[name]["statistic"], shown[name]["p_value"]]
        assert found == pytest.approx([check["statistic"], check["p_value"]]), name
    if not blank:  # with the same runs at each level, Dunnett's intervals are the page's, to scipy's integration
        intervals = shown["dunnett"].confidence_interval(0.95)
        for x, low, high in zip(several["versus"], intervals.low, intervals.high, strict=True):
            assert [low, high] == pytest.approx([x["ci_low"], x["ci_high"]], abs=2e-3), x["level"]


def test_the_best_settings_code_holds_a_factor_of_three_levels_as_the_page_does(api, node):
    catalyst = {"name": "Catalyst", "type": "categorical", "levels": ["A", "B", "C"], "units": ""}
    spec = {"design_type": "full_factorial", "factors": [catalyst, *FACTORS[:2]], "n_center_points": 0}
    held = aimed(held_of(api.ok("api_make_design", spec)), "maximize")
    simulation = api.ok("api_simulate", {**held, "scenario": "main_effects", "sigma": 0.5, "seed": 3})
    request = {**held, "responses": simulation["responses"]}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)
    assert set(best["search"]["held"]) == {"Catalyst_L1", "Catalyst_L2"}, "the catalyst is held at its best level"
    code = best_panel(node, held, simulation["responses"], (analysis, best), simulation)
    level = next(s["value"] for s in best["settings"] if s["name"] == "Catalyst")
    assert f"# The page tried each level and found Catalyst at {level} best" in code
    assert_same_best(run(code), best, held["spec"])


def test_the_code_fits_a_factor_nested_in_another_as_the_page_does(api, node):
    """Supplier + Supplier:Batch, Batch's effect within each supplier: the code adds Supplier's indicators to X."""
    supplier = {"name": "Supplier", "type": "categorical", "levels": ["1", "2", "3"], "units": ""}
    batch = {"name": "Batch", "type": "categorical", "levels": ["a", "b", "c", "d"], "units": ""}
    spec = {"design_type": "full_factorial", "factors": [supplier, batch], "n_replicates": 2, "n_center_points": 0}
    held = aimed(held_of(api.ok("api_make_design", spec)), "maximize")
    sheet = run_sheet(held)
    responses = (sheet["Supplier"].astype(int) * 2 + np.arange(len(sheet)) % 5 * 0.3).tolist()
    model = {"terms": [["Supplier"], ["Supplier", "Batch"]]}
    request = {**held, "responses": responses, "model": model}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)
    code = panel(  # as the page writes it: the analysis's code is given the study's specification
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], True, held["spec"]]),
        ("bestCode", [best, held["spec"], analysis["formula"], True]),
    )
    assert 'X["Supplier_I1"] = (runs["Supplier"] == "1").astype(float)' in code
    namespace = run(code)
    assert_same_fit(namespace, analysis)
    shown = namespace["prediction"]["predictions"][0]
    assert shown["predicted"] == pytest.approx(best["predicted"], abs=1e-6)


# --------------------------------------------------------------------------- what if
def printed_predictions(code: str, names: list[str]) -> dict[str, dict]:
    """Run code as a reader would, and read back each response's prediction it prints: ``Yield {...}``."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})  # noqa: S102 - the page's own code, the subject of this test
    lines = (line.split(" ", 1) for line in out.getvalue().splitlines())
    return {name: ast.literal_eval(rest) for name, rest in lines if name in names}


def what_if_panel(node: str, held: dict, results: dict, reply: dict, settings: dict) -> tuple:
    """Return the what-if panel's code at the settings, and what the page's own JavaScript predicts there."""
    module = source("best.js")
    [columns] = javascript(node, module, [("columnsAt", [reply["factors"], settings])])
    fits = [r for r in reply["responses"] if "terms" in r]
    shown = javascript(node, module, [("predictAt", [fit, columns]) for fit in fits])
    formulas = {fit["name"]: fit["formula"] for fit in fits}
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results]),
        ("whatIfCode", [held["spec"], formulas, columns, settings]),
    )
    return code, dict(zip(formulas, shown, strict=True))


def assert_same_predictions(printed: dict[str, dict], shown: dict[str, dict]) -> None:
    """Assert that the code prints each response's prediction and intervals as the panel shows them."""
    assert set(printed) == set(shown)
    for name, at in shown.items():
        p = printed[name]
        assert p["predicted"] == pytest.approx(at["predicted"], abs=1e-9), name
        assert [p["ci_low"], p["ci_high"]] == pytest.approx(at["confidence"], abs=1e-9), name
        assert [p["pi_low"], p["pi_high"]] == pytest.approx(at["prediction"], abs=1e-9), name


@pytest.mark.slow
def test_the_what_if_code_predicts_each_response_as_the_panel_shows_it(api, node):
    held, simulation = three_responses(api)
    results = by_name(simulation)
    results["Cost"] = [None, *results["Cost"][1:]]  # a run still to do for Cost: its model leaves it out
    request = {**held, **sent_of(results), "models": {"Cost": "interactions"}}
    reply = api.ok("api_predictors", request)
    code, shown = what_if_panel(node, held, results, reply, {"T": 181.5, "P": 1.4, "F": 12.0})
    assert_same_predictions(printed_predictions(code, list(shown)), shown)


def test_the_what_if_code_sets_a_factor_of_several_levels_as_its_contrasts(api, node):
    catalyst = {"name": "Catalyst", "type": "categorical", "levels": ["A", "B", "C"], "units": ""}
    spec = {"design_type": "full_factorial", "factors": [catalyst, FACTORS[0]], "n_replicates": 2, "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    sheet = run_sheet(held)
    y = (50 + sheet["Catalyst"].map({"A": 0.0, "B": 3.0, "C": -3.0}) + (sheet["T"] - 175) / 12.5).tolist()
    y = [round(v + 0.1 * (i % 3), 3) for i, v in enumerate(y)]
    reply = api.ok("api_predictors", {**held, "responses": y, "models": {"y": "interactions"}})
    code, shown = what_if_panel(node, held, {"y": y}, reply, {"Catalyst": "B", "T": 160.0})
    assert '"Catalyst_L1": 0, "Catalyst_L2": 1' in code, "level B as its contrasts"
    assert_same_predictions(printed_predictions(code, list(shown)), shown)


def test_the_what_if_code_predicts_at_each_kept_setting_as_the_list_shows_it(api, node):
    held, simulation = three_responses(api)
    results = by_name(simulation)
    reply = api.ok("api_predictors", {**held, **sent_of(results)})
    module = source("best.js")
    settings = {"Cooler, more pressure": {"T": 160.0, "P": 2.5, "F": 12.0}, "Hot": {"T": 200.0, "P": 1.0, "F": 18.0}}
    kept = [
        {"name": name, "at": javascript(node, module, [("columnsAt", [reply["factors"], one])])[0]}
        for name, one in settings.items()
    ]
    fits = [r for r in reply["responses"] if "terms" in r]
    formulas = {fit["name"]: fit["formula"] for fit in fits}
    [columns] = javascript(node, module, [("columnsAt", [reply["factors"], settings["Hot"]])])
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results]),
        ("whatIfCode", [held["spec"], formulas, columns, settings["Hot"], kept]),
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})  # noqa: S102 - the page's own code, the subject of this test
    printed = dict(line.split(": ", 1) for line in out.getvalue().splitlines() if ", " in line.split(" {", 1)[0])
    for one in kept:
        shown = javascript(node, module, [("predictAt", [fit, one["at"]]) for fit in fits])
        for fit, at in zip(fits, shown, strict=True):
            p = ast.literal_eval(printed[f"{fit['name']}, {one['name']}"])
            assert p["predicted"] == pytest.approx(at["predicted"], abs=1e-9), (fit["name"], one["name"])
            assert [p["pi_low"], p["pi_high"]] == pytest.approx(at["prediction"], abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("transform", [None, "log"], ids=["as measured", "on the log scale"])
def test_the_what_if_code_predicts_each_profile_as_the_panel_draws_it(api, node, transform):
    """The profiles of the first response: each numeric factor moved alone, the others where the panel has them."""
    held, simulation = three_responses(api)
    if transform:
        held = {**held, "spec": {**held["spec"], "analysis_transforms": {"Yield": transform}}}
    results = by_name(simulation)
    reply = api.ok("api_predictors", {**held, **sent_of(results)})
    settings = {"T": 181.5, "P": 1.4, "F": 12.0}
    module = source("best.js")
    [columns] = javascript(node, module, [("columnsAt", [reply["factors"], settings])])
    fits = [r for r in reply["responses"] if "terms" in r]
    formulas = {fit["name"]: fit["formula"] for fit in fits}
    names = [f["name"] for f in reply["factors"]]
    drawn = javascript(node, module, [("profileOf", [fits[0], reply["factors"], settings, name]) for name in names])
    profiled = {"name": fits[0]["name"], "points": len(drawn[0])}
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results]),
        ("whatIfCode", [held["spec"], formulas, columns, settings, [], profiled]),
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})  # noqa: S102 - the page's own code, the subject of this test
    lines = [line.split(": ", 1) for line in out.getvalue().splitlines() if line.startswith("Profile against ")]
    printed = {said.removeprefix("Profile against "): ast.literal_eval(values) for said, values in lines}
    assert list(printed) == names
    for name, profile in zip(names, drawn, strict=True):
        assert printed[name] == pytest.approx([point["predicted"] for point in profile], abs=1e-9), name


def test_the_what_if_code_draws_no_profile_it_was_not_asked_for_nor_one_of_a_factor_with_levels(node):
    catalyst = {"name": "Catalyst", "type": "categorical", "levels": ["A", "B", "C"], "units": ""}
    spec = {"design_type": "full_factorial", "factors": [catalyst, FACTORS[0]], "responses": [{"name": "y"}]}
    at = {"T": 0.5, "Catalyst_L1": 0, "Catalyst_L2": 1}
    settings = {"Catalyst": "B", "T": 187.5}

    def code(*more: object) -> str:
        return javascript(node, source("python.js"), [("whatIfCode", [spec, {"y": "T"}, at, settings, *more])])[0]

    assert "Profile against" not in code()
    assert "import numpy" not in code()
    profiled = code([], {"name": "y", "points": 25})
    assert "for factor in ['T']:" in profiled.replace('"', "'"), "the numeric factor alone"
    assert profiled.startswith("import numpy as np\nimport pandas as pd")
    assert "Profile against" not in code([], {"name": "not fitted", "points": 25})


@pytest.mark.slow
@pytest.mark.parametrize("transform", [None, "log"], ids=["as measured", "on the log scale"])
def test_the_map_s_code_draws_the_surface_the_page_draws(api, node, monkeypatch, transform):
    """The map of the first response over T and F, with P where the dials have it: the same heights, point by point."""
    go = pytest.importorskip("plotly.graph_objects", reason="the page installs plotly for its first chart")
    monkeypatch.setattr(go.Figure, "show", lambda *_, **__: None)  # a reader's own window, which a test has none of
    held, simulation = three_responses(api)
    if transform:
        held = {**held, "spec": {**held["spec"], "analysis_transforms": {"Yield": transform}}}
    results = by_name(simulation)
    reply = api.ok("api_predictors", {**held, **sent_of(results)})
    settings = {"T": 181.5, "P": 1.4, "F": 12.0}
    module = source("best.js")
    fit = next(r for r in reply["responses"] if "terms" in r)
    [columns, drawn] = javascript(
        node,
        module,
        [("columnsAt", [reply["factors"], settings]), ("surfaceOf", [fit, reply["factors"], settings, ["T", "F"]])],
    )
    points = len(drawn["x"])
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results]),
        ("mapCode", [held["spec"], fit["name"], fit["formula"], columns, settings, ["T", "F"], points]),
    )
    made = run(code)
    assert np.asarray(made["z"]) == pytest.approx(np.asarray(drawn["z"]), abs=1e-9)
    (contour,) = made["fig"].data
    assert list(contour.x) == pytest.approx(drawn["x"])
    assert list(contour.y) == pytest.approx(drawn["y"])
    assert (made["fig"].layout.xaxis.title.text, made["fig"].layout.yaxis.title.text) == ("T (degC)", "F (L/min)")


@pytest.mark.slow
def test_the_campaign_s_code_draws_the_runs_and_each_round_s_model_as_the_page_draws_them(api, node, monkeypatch):
    """Two rounds, the second with its ranges moved and its response on the log scale, over T and P."""
    go = pytest.importorskip("plotly.graph_objects", reason="the page installs plotly for its first chart")
    monkeypatch.setattr(go.Figure, "show", lambda *_, **__: None)  # a reader's own window, which a test has none of
    moved = [{**FACTORS[0], "low": 175, "high": 225}, {**FACTORS[1], "low": 0.5, "high": 2.5}, FACTORS[2]]
    pair, drawn, results = ["T", "P"], [], []
    for number, factors in enumerate([FACTORS, moved], start=1):
        held = held_of(api.ok("api_make_design", {"design_type": "box_behnken", "factors": factors}))
        if number == 2:
            held = {**held, "spec": {**held["spec"], "analysis_transforms": {"y": "log"}}}
        values = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": number})["responses"]
        reply = api.ok("api_predictors", {**held, "responses": values})
        (fit,) = (r for r in reply["responses"] if "terms" in r)
        middle = {f["name"]: (f["low"] + f["high"]) / 2 for f in reply["factors"]}
        at, surface = javascript(
            node,
            source("best.js"),
            [("columnsAt", [reply["factors"], middle]), ("surfaceOf", [fit, reply["factors"], middle, pair])],
        )
        sheet = [dict(zip(held["columns"], row, strict=True)) for row in held["rows"]]
        ranges = [[f["low"], f["high"]] for f in factors[:2]]
        x, y = [row["T"] for row in sheet], [row["P"] for row in sheet]
        drawn.append({"round": number, "x": x, "y": y, "fit": fit, "at": at, "ranges": ranges, "surface": surface})
        results += [{"round": number, "run": run_, "value": value} for run_, value in enumerate(values)]

    axes = [{"name": "T", "units": "degC"}, {"name": "P", "units": "bar"}]
    words = {"label": "Yield", "units": "%", "pair": pair, "axes": axes, "points": len(drawn[0]["surface"]["x"])}
    (code,) = javascript(node, source("python.js"), [("campaignMapCode", [drawn, words])])
    fig = run(code)["fig"]
    contours = [trace for trace in fig.data if trace.type == "contour"]
    assert len(contours) == 2
    for contour, one in zip(contours, drawn, strict=True):
        assert np.asarray(contour.z) == pytest.approx(np.asarray(one["surface"]["z"]), rel=1e-9), one["round"]
        assert list(contour.x) == pytest.approx(one["surface"]["x"])  # over its own ranges, and no further
        assert list(contour.y) == pytest.approx(one["surface"]["y"])
    marks = [trace for trace in fig.data if trace.type == "scatter"]
    assert [(trace.name, list(trace.x)) for trace in marks] == [(f"Round {one['round']}", one["x"]) for one in drawn]
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ("T (degC)", "P (bar)")

    # Every run in the order made, and the best so far for each goal as campaign.js works it out.
    for goal in ({"goal": "maximize"}, {"goal": "minimize"}, {"goal": "target", "target": "64.2"}, {"goal": "learn"}):
        (best,) = javascript(node, source("campaign.js"), [("bestSoFar", [results, goal])])
        (code,) = javascript(node, source("python.js"), [("campaignRunsCode", [results, {"label": "Yield", **goal}])])
        made = run(code)
        assert made["runs"]["Yield"].tolist() == [one["value"] for one in results]
        assert made["runs"]["Round"].tolist() == [one["round"] for one in results]
        if goal["goal"] == "learn":
            assert "Best so far" not in made["runs"], "a goal that names no best has no line of it"
        else:
            assert made["runs"]["Best so far"].tolist() == pytest.approx(best), goal


# --------------------------------------------------------------------------- the code of each figure
def figure_drawn(node, monkeypatch, steps: tuple, options: dict, model: str, complete: bool, spec: dict):
    """Return the figure that a figure's code draws, run after the code of the steps it builds on."""
    go = pytest.importorskip("plotly.graph_objects", reason="the page installs plotly for its first chart")
    monkeypatch.setattr(go.Figure, "show", lambda *_, **__: None)  # a reader's own window, which a test has none of
    code = panel(node, *steps, ("figureCode", ["A figure", options, model, complete, spec]))
    return run(code)["figure"]


def assert_same_figure(drawn, shown: dict, key: str, values: tuple[str, ...] = ("y", "z")) -> None:
    """Assert that the library's figure holds the numbers of the page's: trace for trace, what each one draws."""
    theirs_drawn = [trace for trace in shown["data"] if trace.get("meta") != "interval"]  # the page's own bands
    assert len(drawn.data) == len(theirs_drawn), key
    for mine, theirs in zip(drawn.data, theirs_drawn, strict=True):
        assert mine.type == theirs["type"], key
        for name in values:
            if theirs.get(name) is not None and mine.type != "scatter3d":
                np.testing.assert_allclose(
                    np.asarray(mine[name], dtype=float), np.asarray(theirs[name], dtype=float), atol=1e-9, err_msg=key
                )
        if theirs.get("text") is not None:  # the page names each run at the head of its words
            words = [re.sub(r"^Run \d+<br>", "", said) for said in theirs["text"]]
            assert list(mine.text) == words, f"{key}: the numbers printed on it"


@pytest.mark.slow
def test_each_figure_s_code_draws_the_figure_the_page_draws(api, node, monkeypatch):
    """The window of each figure shows the code that draws it with the library: the same numbers, in coded units."""
    demo = api.ok("api_catalogue")["demo"]
    held = held_of(api.ok("api_make_design", demo["design"]))
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulation["responses"]})
    plots = api.ok("api_plots", {**held, "responses": simulation["responses"], "truth": simulation["truth"]})
    figures = {key: json.loads(text) for key, text in plots["figures"].items()}
    steps = (
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, simulation["responses"]]),
        ("analyseCode", [analysis["formula"], True]),
    )
    assert set(plots["plot_options"]) == set(figures) - {"true_contour"}, "each figure drawn from the analysis"
    for key, options in plots["plot_options"].items():
        drawn = figure_drawn(node, monkeypatch, steps, options, analysis["formula"], True, held["spec"])
        assert_same_figure(drawn, figures[key], key)
    assert plots["plot_options"]["contour"] == {"plot_type": "contour", "factors_to_plot": ["Temperature", "Pressure"]}
    assert plots["plot_options"]["residuals"] == {"plot_type": "residuals_vs_fitted"}


@pytest.mark.slow
def test_a_surface_s_code_holds_the_other_factors_where_the_page_holds_them(api, node, monkeypatch):
    """With the best settings found, the page draws its surfaces through them: the code holds the others there too."""
    demo = api.ok("api_catalogue")["demo"]
    held = held_of(api.ok("api_make_design", demo["design"]))
    responses = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    plots = api.ok("api_plots", {**held, "responses": responses, "best": EXAMPLE_BEST})
    steps = (
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], True]),
    )
    for key in ("contour", "surface_3d", "prediction_variance"):
        options = plots["plot_options"][key]
        assert options["hold_values"] == {"Flow": 0.6}, key
        drawn = figure_drawn(node, monkeypatch, steps, options, analysis["formula"], True, held["spec"])
        assert_same_figure(drawn, json.loads(plots["figures"][key]), key)
    assert "hold_values" not in plots["plot_options"]["perturbation"], "each factor moves, the others at their center"


@pytest.mark.slow
def test_a_figure_s_code_leaves_out_the_runs_still_to_do_and_fits_on_the_scale_the_page_fits_on(api, node, monkeypatch):
    held = held_of(api.ok("api_make_design", {"design_type": "ccd", "factors": FACTORS}))
    held = {**held, "spec": {**held["spec"], "analysis_transforms": {held["spec"]["response"]: "log"}}}
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 2})
    responses = [None, *simulation["responses"][1:-1], None]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    plots = api.ok("api_plots", {**held, "responses": responses})
    steps = (
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], False, held["spec"]]),
    )
    figure = functools.partial(figure_drawn, node, monkeypatch, steps)
    residuals = figure(plots["plot_options"]["residuals"], analysis["formula"], False, held["spec"])
    assert_same_figure(residuals, json.loads(plots["figures"]["residuals"]), "residuals", values=("x", "y"))
    assert len(residuals.data[0].y) == len(responses) - 2, "a residual for each run that has a result"
    # The page puts a prediction back in the response's units; the library draws it on the scale it is fitted on.
    effect = figure(plots["plot_options"]["perturbation"], analysis["formula"], False, held["spec"])
    shown = [t for t in json.loads(plots["figures"]["perturbation"])["data"] if t.get("meta") != "interval"]
    for mine, theirs in zip(effect.data, shown, strict=True):
        np.testing.assert_allclose(np.exp(np.asarray(mine.y, dtype=float)), theirs["y"], rtol=1e-9)


@pytest.mark.slow
def test_the_half_normal_plot_s_code_judges_the_effects_by_lenth_s_margins_as_the_page_does(api, node, monkeypatch):
    spec = {"design_type": "plackett_burman", "factors": factors_named(*"ABCDEFG"), "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    responses = api.ok("api_simulate", {**held, "scenario": "main_effects", "seed": 1})["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    plots = api.ok("api_plots", {**held, "responses": responses})
    options = plots["plot_options"]["half_normal"]
    assert options == {"plot_type": "half_normal"}
    steps = (
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], True]),
    )
    drawn = figure_drawn(node, monkeypatch, steps, options, analysis["formula"], True, held["spec"])
    assert_same_figure(drawn, json.loads(plots["figures"]["half_normal"]), "half_normal", values=("x", "y"))
    margins = {shape["y0"] for shape in drawn.layout.shapes}
    assert margins == {shape["y0"] for shape in json.loads(plots["figures"]["half_normal"])["layout"]["shapes"]}


def test_at_a_level_of_the_study_s_own_the_half_normal_plot_s_code_draws_the_page_s_margins(api, node, monkeypatch):
    """Lenth's margins move with the level: at 0.01 the code names it and draws the page's wider margins."""
    go = pytest.importorskip("plotly.graph_objects", reason="the page installs plotly for its first chart")
    monkeypatch.setattr(go.Figure, "show", lambda *_, **__: None)  # a reader's own window, which a test has none of
    spec = {"design_type": "plackett_burman", "factors": factors_named(*"ABCDEFG"), "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    responses = api.ok("api_simulate", {**held, "scenario": "main_effects", "seed": 1})["responses"]
    request = {**held, "responses": responses, "significance_level": 0.01}
    analysis = api.ok("api_analyze_runs", request)
    figure = {level: api.ok("api_plots", {**request, "significance_level": level}) for level in (0.05, 0.01)}
    options = figure[0.01]["plot_options"]["half_normal"]
    steps = [
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], True]),
        ("figureCode", ["A figure", options, analysis["formula"], True, held["spec"]]),
    ]
    at_strict = source("python.js") + '\nimport { setSignificance } from "./numbers.js";\nsetSignificance(0.01);\n'
    code = javascript(node, source("python.js"), [("cumulative", javascript(node, at_strict, steps))])[0]
    drawn = run(code)["figure"]
    margins = {
        level: {s["y0"] for s in json.loads(f["figures"]["half_normal"])["layout"]["shapes"]}
        for level, f in figure.items()
    }
    assert {shape["y0"] for shape in drawn.layout.shapes} == margins[0.01]
    assert max(margins[0.01]) > max(margins[0.05]), "the margins at 0.01 are the wider"


@pytest.mark.slow
def test_a_figure_s_code_for_a_study_with_limits_draws_the_surface_over_every_setting(api, node, monkeypatch):
    """The page shades the settings the limits rule out; the code says it does not, and draws the same surface."""
    design, responses = _limited(api, "T + 25 * P <= 250")
    held = held_of(design)
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    plots = api.ok("api_plots", {**held, "responses": responses})
    assert plots["shaded"]
    steps = (
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], True]),
    )
    options = plots["plot_options"]["contour"]
    assert options == {"plot_type": "contour", "factors_to_plot": ["T", "P"]}, "the region is not one of them"
    code = panel(node, *steps, ("figureCode", ["A figure", options, analysis["formula"], True, held["spec"]]))
    assert "# The page also shades the settings your limits rule out" in code
    drawn = figure_drawn(node, monkeypatch, steps, options, analysis["formula"], True, held["spec"])
    shown = json.loads(plots["figures"]["contour"])["data"]
    assert [trace["name"] for trace in shown if trace["name"] not in {t.name for t in drawn.data}] == ["Cannot be run"]
    np.testing.assert_allclose(
        np.asarray(drawn.data[0].z, dtype=float), np.asarray(shown[0]["z"], dtype=float), atol=1e-9
    )


def test_a_surface_s_code_draws_it_in_the_study_s_colour_map(api, node, monkeypatch):
    """The library draws a surface in batlow, the page in the study's map: the code it shows redraws it so (#1381)."""
    go = pytest.importorskip("plotly.graph_objects", reason="the page installs plotly for its first chart")
    monkeypatch.setattr(go.Figure, "show", lambda *_, **__: None)  # a reader's own window, which a test has none of
    held = held_of(api.ok("api_make_design", {"design_type": "ccd", "alpha": "face_centered", "factors": FACTORS}))
    responses = [float(i % 5) + 0.1 * i for i in range(len(held["rows"]))]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    options = api.ok("api_plots", {**held, "responses": responses})["plot_options"]["contour"]
    steps = [
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], True]),
        ("figureCode", ["A figure", options, analysis["formula"], True, held["spec"]]),
    ]
    chosen = source("python.js") + '\nexport { setColourMap } from "./charts.js";\n'
    _, *codes = javascript(node, chosen, [("setColourMap", ["devon"]), *steps])
    figure = run(javascript(node, source("python.js"), [("cumulative", codes)])[0])["figure"]
    surfaces = [trace for trace in figure.data if "colorscale" in trace and trace.colorscale]
    assert surfaces, "the figure of the fitted response draws one"
    for trace in surfaces:
        assert [list(stop) for stop in trace.colorscale] == colorscale("devon")


# --------------------------------------------------------------------------- the page's own figures
def own_figure(node, monkeypatch, steps: tuple, key: str, spec: dict, about: dict | None = None):
    """Return the figure that the code of one of the page's own figures draws, run after the steps it builds on."""
    go = pytest.importorskip("plotly.graph_objects", reason="the page installs plotly for its first chart")
    monkeypatch.setattr(go.Figure, "show", lambda *_, **__: None)  # a reader's own window, which a test has none of
    code = panel(node, *steps, ("ownFigureCode", [key, "A figure", spec, about or {}]))
    return run(code)["figure"]


def steps_of(held: dict, responses: list, analysis: dict, simulation: dict | None = None) -> tuple:
    """Return the three steps a figure's code builds on: the design, its results and their analysis."""
    results = [held, responses] if simulation is None else [held, responses, simulation]
    return (
        ("designCode", [held["spec"]]),
        ("resultsCode", results),
        ("analyseCode", [analysis["formula"], None not in responses, held["spec"]]),
    )


@pytest.mark.slow
def test_the_coefficients_code_draws_each_estimate_with_its_interval(api, node, monkeypatch):
    demo = api.ok("api_catalogue")["demo"]
    held = held_of(api.ok("api_make_design", demo["design"]))
    responses = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    drawn = own_figure(node, monkeypatch, steps_of(held, responses, analysis), "coefficients", held["spec"]).data[0]
    shown = [c for c in analysis["coefficients"] if c["term"] != "Intercept"]
    assert list(drawn.y) == [c["term"] for c in shown], "a row a term, the intercept left out as on the page"
    np.testing.assert_allclose(drawn.x, [c["coefficient"] for c in shown])
    np.testing.assert_allclose(drawn.error_x.array, [c["ci_high"] - c["coefficient"] for c in shown])
    np.testing.assert_allclose(drawn.error_x.arrayminus, [c["coefficient"] - c["ci_low"] for c in shown])


@pytest.mark.slow
def test_the_coefficients_code_draws_a_fit_that_uses_every_run_with_no_interval(api, node, monkeypatch):
    spec = {"design_type": "plackett_burman", "factors": factors_named(*"ABCDEFG"), "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    responses = api.ok("api_simulate", {**held, "scenario": "main_effects", "seed": 1})["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    drawn = own_figure(node, monkeypatch, steps_of(held, responses, analysis), "coefficients", held["spec"]).data[0]
    np.testing.assert_allclose(
        drawn.x, [c["coefficient"] for c in analysis["coefficients"] if c["term"] != "Intercept"]
    )
    assert not np.nansum(np.asarray(drawn.error_x.array, dtype=float)), "nothing is left to say how well each is known"


@pytest.mark.slow
def test_the_levels_code_draws_every_run_at_its_level_and_each_level_s_average(api, node, monkeypatch):
    spec = {"design_type": "full_factorial", "factors": [SUPPLIER], "n_replicates": 5, "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    simulated = api.ok("api_simulate", {**held, "scenario": "main_effects", "sigma": 1, "seed": 2})["responses"]
    responses = [None, *simulated[1:]]  # a run still to do is left out, as the page leaves it out
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    runs, averages = own_figure(node, monkeypatch, steps_of(held, responses, analysis), "levels", held["spec"]).data
    levels = analysis["comparison"]["levels"]
    assert sorted(runs.y) == sorted(r["value"] for level in levels for r in level["runs"])
    assert len(runs.y) == len(responses) - 1
    assert dict(zip(averages.x, averages.y, strict=True)) == pytest.approx(
        {str(level["level"]): level["mean"] for level in levels}
    )


@pytest.mark.slow
@pytest.mark.parametrize("best", [None, EXAMPLE_BEST], ids=["at the center", "at the best settings"])
def test_the_true_surface_s_code_draws_the_process_the_results_were_simulated_from(api, node, monkeypatch, best):
    demo = api.ok("api_catalogue")["demo"]
    held = held_of(api.ok("api_make_design", demo["design"]))
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    responses = simulation["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    sent = {**held, "responses": responses, "truth": simulation["truth"], **({"best": best} if best else {})}
    plots = api.ok("api_plots", sent)
    about = {"contour": plots["plot_options"]["contour"], "process": simulation}
    drawn = own_figure(node, monkeypatch, steps_of(held, responses, analysis), "true_contour", held["spec"], about)
    surface = json.loads(plots["figures"]["true_contour"])["data"][0]
    assert (about["contour"].get("hold_values") is not None) == (best is not None)
    for name in ("x", "y", "z"):
        np.testing.assert_allclose(np.asarray(drawn.data[0][name], dtype=float), surface[name], atol=1e-9, err_msg=name)
    # With results typed over the simulated ones, step 2 makes no process: the figure's code makes its own.
    typed = steps_of(held, [round(value) for value in responses], analysis)
    again = own_figure(node, monkeypatch, typed, "true_contour", held["spec"], about)
    np.testing.assert_allclose(np.asarray(again.data[0].z, dtype=float), surface["z"], atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("request_", [MIXTURES[1], MIXTURES[4]], ids=["lattice", "vertices"])
def test_a_mixture_s_triangles_are_drawn_by_their_code_over_the_page_s_blends(api, node, monkeypatch, request_):
    """The fitted response and the true one, over the blends the bounds allow, with the runs on the fitted one."""
    held = held_of(api.ok("api_make_design", {**request_, "factors": components(request_["design_type"])}))
    simulation = api.ok("api_simulate", {**held, "seed": 5})
    responses = simulation["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    ternary = api.ok("api_plots", {**held, "responses": responses, "truth": simulation["truth"]})["ternary"]
    steps = steps_of(held, responses, analysis, simulation)
    for key, values in (("ternary", "fitted"), ("true_ternary", "true")):
        surface, runs = own_figure(node, monkeypatch, steps, key, held["spec"], {"process": simulation}).data
        for corner in "abc":
            np.testing.assert_allclose(surface[corner], ternary["grid"][corner], atol=1e-9, err_msg=f"{key} {corner}")
            np.testing.assert_allclose(runs[corner], ternary["runs"][corner], atol=1e-12)
        np.testing.assert_allclose(surface.marker.color, ternary["grid"][values], rtol=1e-5, err_msg=key)


@pytest.mark.slow
@pytest.mark.parametrize("through", ["the reference blend", "the best blend"])
def test_the_cox_traces_code_moves_each_component_as_the_page_moves_it(api, node, monkeypatch, through):
    four = [{"name": name, "low": None, "high": None, "units": ""} for name in "ABCD"]
    made = held_of(api.ok("api_make_design", {"design_type": "simplex_lattice", "factors": four}))
    held = aimed(made, "maximize")
    responses = api.ok("api_simulate", {**held, "seed": 5})["responses"]
    request = {**held, "responses": responses}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)["coded"] if through == "the best blend" else None
    cox = api.ok("api_plots", {**request, **({"best": best} if best else {})})["cox"]
    assert (cox.get("through") == "best") == (best is not None)
    drawn = own_figure(node, monkeypatch, steps_of(held, responses, analysis), "cox", held["spec"], {"cox": cox})
    assert [trace.name for trace in drawn.data] == [trace["component"] for trace in cox["traces"]]
    for mine, theirs in zip(drawn.data, cox["traces"], strict=True):
        np.testing.assert_allclose(mine.x, theirs["change"], atol=1e-6, err_msg=theirs["component"])
        np.testing.assert_allclose(mine.y, theirs["fitted"], rtol=1e-5, err_msg=theirs["component"])


@pytest.mark.slow
@pytest.mark.parametrize("through", ["the reference blend", "the best blend"])
def test_the_triangle_of_three_of_four_components_is_drawn_by_its_code_as_the_page_draws_it(
    api, node, monkeypatch, through
):
    four = [{"name": name, "low": None, "high": None, "units": ""} for name in "ABCD"]
    made = held_of(api.ok("api_make_design", {"design_type": "simplex_lattice", "factors": four}))
    held = aimed(made, "maximize")
    responses = api.ok("api_simulate", {**held, "seed": 5})["responses"]
    request = {**held, "responses": responses}
    analysis = api.ok("api_analyze_runs", request)
    best = api.ok("api_best_settings", request)["coded"] if through == "the best blend" else None
    ternary = api.ok("api_plots", {**request, "triangle": ["A", "C", "D"], **({"best": best} if best else {})})[
        "ternary"
    ]
    about = {"ternary": ternary}
    drawn = own_figure(node, monkeypatch, steps_of(held, responses, analysis), "ternary", held["spec"], about)
    (surface,) = drawn.data  # no runs: a run of four components is not on this triangle
    for corner in "abc":
        np.testing.assert_allclose(surface[corner], ternary["grid"][corner], atol=1e-9, err_msg=corner)
    np.testing.assert_allclose(surface.marker.color, ternary["grid"]["fitted"], rtol=1e-5)
    assert drawn.layout.ternary.sum == pytest.approx(ternary["sum"], abs=1e-9)


def test_a_figure_the_page_could_not_have_drawn_has_no_code(node):
    spec = {"response": "y", "factors": factors_named("T", "P")}
    asked = [
        ("ownFigureCode", ["true_contour", "A figure", spec, {}]),
        ("ownFigureCode", ["true_ternary", "A figure", spec, {}]),
        ("ownFigureCode", ["cox", "A figure", spec, {}]),
        ("ownFigureCode", ["contour", "A figure", spec, {}]),
    ]
    assert javascript(node, source("python.js"), asked) == [None] * 4, (
        "no process, no traces, or the library's own figure"
    )


@pytest.mark.slow
def test_the_notebook_draws_each_figure_after_the_analysis_it_is_of(api, node, monkeypatch):
    """Each figure's code is a cell of its own, under its title, so a notebook run top to bottom shows them all."""
    go = pytest.importorskip("plotly.graph_objects", reason="the page installs plotly for its first chart")
    shown: list = []
    monkeypatch.setattr(go.Figure, "show", lambda figure, *_, **__: shown.append(figure))
    demo = api.ok("api_catalogue")["demo"]
    held = held_of(api.ok("api_make_design", demo["design"]))
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    responses = simulation["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    plots = api.ok("api_plots", {**held, "responses": responses, "truth": simulation["truth"]})
    spec, formula = held["spec"], analysis["formula"]
    about = {"contour": plots["plot_options"]["contour"], "process": simulation}
    codes = [
        {"title": "Coefficients", "code": page_code(node, "ownFigureCode", "coefficients", "Coefficients", spec)},
        {
            "title": "Fitted",
            "code": page_code(node, "figureCode", "Fitted", plots["plot_options"]["contour"], formula, True, spec),
        },
        {"title": "True", "code": page_code(node, "ownFigureCode", "true_contour", "True", spec, about)},
    ]
    book = notebook_of(
        node,
        design=held,
        responses=responses,
        sources=["simulated"] * len(responses),
        simulation=simulation,
        model=formula,
        figures={"response": spec["response"], "codes": codes},
    )
    nbformat = pytest.importorskip("nbformat", reason="the notebook is checked against Jupyter's own schema")
    nbformat.validate(nbformat.reads(json.dumps(book), as_version=4))
    cells = [(cell["cell_type"], "".join(cell["source"])) for cell in book["cells"]]
    assert [text for kind, text in cells if kind == "markdown"][-3:] == ["### Coefficients", "### Fitted", "### True"]
    code = [text for kind, text in cells if kind == "code"]
    assert code[-3:] == [figure["code"] for figure in codes], "each after the analysis, in the page's order"
    namespace: dict = {}
    for cell in code[1:]:
        run(cell, namespace)
    assert len(shown) == 3
    other = notebook_of(
        node,
        design=held,
        responses=responses,
        sources=["simulated"] * len(responses),
        simulation=simulation,
        model=formula,
        figures={"response": "another response", "codes": codes},
    )
    assert len(other["cells"]) == len(book["cells"]) - 6, (
        "figures of a response the notebook does not analyse are left out"
    )


# --------------------------------------------------------------------------- the figures beyond the analysis's charts
def drawn_by(monkeypatch, code: str):
    """Return the figure a figure's whole code draws, run as a reader would run it."""
    go = pytest.importorskip("plotly.graph_objects", reason="the page installs plotly for its first chart")
    monkeypatch.setattr(go.Figure, "show", lambda *_, **__: None)  # a reader's own window, which a test has none of
    return run(code)


@pytest.mark.slow
def test_the_pareto_plot_s_code_draws_each_coefficient_s_size_with_its_interval(api, node, monkeypatch):
    demo = api.ok("api_catalogue")["demo"]
    held = held_of(api.ok("api_make_design", demo["design"]))
    responses = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    bars = own_figure(node, monkeypatch, steps_of(held, responses, analysis), "pareto", held["spec"]).data[0]
    shown = sorted(
        (c for c in analysis["coefficients"] if c["term"] != "Intercept"), key=lambda c: -abs(c["coefficient"])
    )
    assert list(bars.y) == [c["term"] for c in shown], "the largest first"
    np.testing.assert_allclose(bars.x, [abs(c["coefficient"]) for c in shown], atol=1e-9)
    np.testing.assert_allclose(bars.error_x.array, [c["ci_high"] - c["coefficient"] for c in shown], atol=1e-9)
    assert list(bars.marker.color) == ["#E69F00" if c["coefficient"] > 0 else "#0072B2" for c in shown]


def test_the_effects_view_s_code_draws_each_effect_over_its_range_and_the_change_worth_finding(api, node, monkeypatch):
    demo = api.ok("api_catalogue")["demo"]
    held = held_of(api.ok("api_make_design", demo["design"]))
    responses = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    figure = own_figure(node, monkeypatch, steps_of(held, responses, analysis), "effects", held["spec"], {"change": 2})
    bars = figure.data[0]
    span = lambda c: 1 if c["term"].startswith("I(") else 2  # noqa: E731 - a square from the center to an end
    shown = sorted(
        (c for c in analysis["coefficients"] if c["term"] != "Intercept"),
        key=lambda c: -abs(span(c) * c["coefficient"]),
    )
    assert any(span(c) == 1 for c in shown), "the worked example's model has squares"
    assert list(bars.y) == [c["term"] for c in shown], "the largest effect first"
    np.testing.assert_allclose(bars.x, [span(c) * c["coefficient"] for c in shown], atol=1e-9)
    np.testing.assert_allclose(
        bars.error_x.array, [span(c) * (c["ci_high"] - c["coefficient"]) for c in shown], atol=1e-9
    )
    assert list(bars.marker.color) == ["#E69F00" if c["coefficient"] > 0 else "#0072B2" for c in shown]
    (band,) = figure.layout.shapes
    assert (band.x0, band.x1) == (-2, 2), "the change worth finding, either side of no change"
    assert figure.layout.xaxis.title.text == "Change in Yield (%) over the range"


@pytest.mark.slow
def test_the_pareto_plot_s_code_judges_a_fit_of_every_run_by_lenth_s_margins(api, node, monkeypatch):
    spec = {"design_type": "plackett_burman", "factors": factors_named(*"ABCDEFG"), "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    responses = api.ok("api_simulate", {**held, "scenario": "main_effects", "seed": 1})["responses"]
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    assert analysis["lenth"], "Lenth's margins, which the page draws"
    assert analysis["coefficients"][1]["ci_high"] is None, "no run is left to judge by"
    about = {"lenth": True, "model": analysis["formula"], "complete": True}
    figure = own_figure(node, monkeypatch, steps_of(held, responses, analysis), "pareto", held["spec"], about)
    assert figure.data[0].error_x.array is None
    margins = [shape.x0 for shape in figure.layout.shapes]
    np.testing.assert_allclose(margins, [analysis["lenth"]["ME"] / 2, analysis["lenth"]["SME"] / 2], rtol=1e-9)


def same_map(heatmap, terms: list[str], matrix: list[list]) -> None:
    """Assert that a map drawn by the code holds the page's: its terms in order, and each pair's correlation."""
    assert list(heatmap.x) == list(heatmap.y) == terms
    np.testing.assert_allclose(
        np.asarray(heatmap.z, dtype=float), np.asarray(matrix, dtype=float), atol=1e-9, equal_nan=True
    )


@pytest.mark.slow
def test_the_design_s_correlation_map_code_draws_the_page_s_map(api, node, monkeypatch):
    spec = {"design_type": "fractional_factorial", "factors": factors_named(*"ABCDE")}
    held = held_of(api.ok("api_make_design", spec))
    found = api.ok("api_correlations", held)
    to_do = [None] * len(held["rows"])  # the map is of the runs: it is drawn before any has a result
    steps = (("designCode", [held["spec"]]), ("resultsCode", [held, to_do]), ("correlationCode", [list("ABCDE")]))
    same_map(drawn_by(monkeypatch, panel(node, *steps))["figure"].data[0], found["terms"], found["matrix"])


@pytest.mark.slow
@pytest.mark.parametrize("limits", [[], LIMITS[:1]], ids=["the cube", "inside limits"])
def test_the_comparison_s_figures_code_draws_the_page_s_curves_and_maps(api, node, monkeypatch, limits):
    made, held_designs = generate_all(api, FACTORS, "main_quadratic", limits)
    compared = api.ok("api_compare", {"model": "main_quadratic", "designs": held_designs})
    designs = [
        {"label": m["label"], "design_type": m["design_type"], "options": m["design"]["spec"]["options"]} for m in made
    ]
    stored = made[0]["design"]["spec"].get("constraints", [])
    compare = ("compareCode", [FACTORS, designs, "main_quadratic", stored])
    fds = panel(node, compare, ("comparisonFigureCode", [{"asked": compared["fds_asked"], "limits": stored}]))
    curves = drawn_by(monkeypatch, fds)["figure"].data
    assert [curve.name for curve in curves] == list(compared["fds"]), "a curve for each design that fits the model"
    for curve in curves:
        np.testing.assert_allclose(curve.x, compared["fds"][curve.name]["fraction"], atol=1e-12)
        np.testing.assert_allclose(curve.y, compared["fds"][curve.name]["variance"], rtol=1e-9)
    label = compared["designs"][-1]
    one = panel(node, compare, ("comparisonFigureCode", [{"label": label}]))
    theirs = compared["correlations"][label]
    same_map(drawn_by(monkeypatch, one)["figure"].data[0], theirs["terms"], theirs["matrix"])


@pytest.mark.slow
@pytest.mark.parametrize("transform", [None, "log"])
def test_the_code_of_how_tightly_to_hold_each_factor_moves_each_as_the_page_does(
    api, node, monkeypatch, worked_example, transform
):
    held, responses = aimed(worked_example["held"], "maximize"), worked_example["simulated"]["responses"]
    if transform:
        held = {**held, "spec": {**held["spec"], "analysis_transforms": {held["spec"]["response"]: transform}}}
    analysis = api.ok("api_analyze_runs", {**held, "responses": responses})
    best = api.ok("api_best_settings", {**held, "responses": responses})
    moved = [s for s in best["sensitivity"] if s["in_model"]]
    names = [s["name"] for s in moved]
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("analyseCode", [analysis["formula"], True, held["spec"]]),
        ("bestCode", [best, held["spec"], analysis["formula"], True]),
        ("sensitivityCode", [names, held["spec"], analysis["formula"], True]),
    )
    lowered, raised = drawn_by(monkeypatch, code)["figure"].data
    assert list(lowered.y) == list(raised.y) == names
    np.testing.assert_allclose(lowered.x, [s["down"] for s in moved], atol=1e-6)
    np.testing.assert_allclose(raised.x, [s["up"] for s in moved], atol=1e-6)


@pytest.mark.slow
def test_the_overlay_s_code_draws_each_response_s_surface_and_boundary_as_the_page_does(api, node, monkeypatch):
    held, simulation = three_responses(api)
    goals = {"Yield": {"importance": 5, "shape": "more"}, "Cost": {"high": 70.0}}
    request = {**held, **sent_of(by_name(simulation)), "models": {"Cost": "interactions"}, "goals": goals}
    balance = api.ok("api_balance", request)
    overlay = balance["overlay"]
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, by_name(simulation), simulation]),
        ("balanceCode", [balance, held["spec"]]),
        ("overlayCode", [overlay, held["spec"]]),
    )
    namespace = drawn_by(monkeypatch, code)
    drawn = {trace.name: np.asarray(trace.z, dtype=float) for trace in namespace["figure"].data}
    corners = np.ix_([0, -1], [0, -1])  # the page's grid is finer than the library's: the two share their corners
    for surface in overlay["responses"]:
        np.testing.assert_allclose(drawn[surface["name"]][corners], np.asarray(surface["z"])[corners], rtol=1e-5)
    # Each response is bounded where the page draws its boundary, and the others are held at the balanced settings.
    bounded = {r["name"]: [r[end] for end in ("low", "high") if end in r] for r in namespace["responses"]}
    assert bounded == {surface["name"]: surface["levels"] for surface in overlay["responses"]}
    assert [overlay["x"], overlay["y"]] == ["T", "P"]
    held_at = {h["name"]: h["value"] for h in overlay["held"]}
    assert namespace["balanced"]["optimal_actual"]["F"] == pytest.approx(held_at["F"], abs=1e-6)
    assert drawn["Sweet spot"].max() == 1, "where every response is acceptable"


@pytest.mark.slow
def test_the_what_if_curve_s_code_predicts_the_curve_the_panel_draws(api, node, monkeypatch, worked_example):
    held = curved(worked_example["held"])
    fit = api.ok("api_functional", held)
    at = {"T": 0.6, "P": -0.5, "F": 0.2}  # the panel's settings, as the columns of X
    design = {key: held[key] for key in ("spec", "columns", "rows")}
    to_do = [None] * len(held["rows"])  # the study's only results are its curves
    steps = (
        ("designCode", [held["spec"]]),
        ("resultsCode", [design, to_do]),
        ("curveCode", [design, held["curves"], fit, at]),
    )
    namespace = drawn_by(monkeypatch, panel(node, *steps))
    row = np.array([np.prod([at[name] for name in parts]) for parts in fit["terms"]])
    shown = np.asarray(fit["mean"]) + row @ np.asarray(fit["map"])  # as the page draws it: best.js curveAt
    np.testing.assert_allclose(namespace["predicted"]["curve"].iloc[0], shown, rtol=1e-6, atol=1e-6)
    assert namespace["predicted"]["n_bootstrap"] == len(fit["replicates"]), "as many replicates as the page's bands"
    high, low, predicted = namespace["figure"].data
    assert list(predicted.x) == fit["points"] == held["spec"]["curve"]["argument"]["values"]
    assert np.all(np.asarray(high.y) >= np.asarray(predicted.y))
    assert np.all(np.asarray(low.y) <= np.asarray(predicted.y))


@pytest.mark.slow
@pytest.mark.parametrize("limits", [{}, {"Purity": [62, 64], "Cost": [None, 62]}], ids=["every setting", "limited"])
def test_the_trade_off_chart_s_code_tries_the_page_s_settings_and_draws_its_line(api, node, monkeypatch, limits):
    held, simulation = three_responses(api)
    request = {**held, **sent_of(by_name(simulation)), "models": {"Cost": "interactions"}}
    balance = api.ok("api_balance", request)
    tried = api.ok("api_trade_offs", {**request, "at": balance["coded"]})
    view = {"asked": tried["asked"], "across": "Yield", "up": "Cost", "colour": "Purity", "limits": limits}
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, by_name(simulation), simulation]),
        ("balanceCode", [balance, held["spec"]]),
        ("tradeOffCode", [view, balance, held["spec"]]),
    )
    namespace = drawn_by(monkeypatch, code)
    candidates = tried["candidates"]
    for name, shown in candidates["responses"].items():  # the page rounds each to six figures
        np.testing.assert_allclose(namespace["cloud"][name], shown, rtol=1e-5)
    np.testing.assert_allclose(175 + 25 * namespace["tried"]["T"], candidates["settings"]["T"], rtol=1e-5)
    # The page's own line, from web/best.js, over the settings that meet the limits.
    ends = {name: [None, None] for name in candidates["responses"]} | limits
    meets = javascript(node, source("best.js"), [("meetsLimits", [candidates["responses"], ends])])[0]
    assert namespace["kept"].index.tolist() == [i for i, met in enumerate(meets) if met]
    across = [-v for v in candidates["responses"]["Yield"]]
    front = javascript(node, source("best.js"), [("frontier", [across, candidates["responses"]["Cost"], meets])])[0]
    assert namespace["front"].tolist() == front
    far = np.asarray(candidates["far"], dtype=bool)
    assert far.any(), "some settings tried lie beyond the runs"
    np.testing.assert_array_equal(namespace["far"], far)
    guessed, cloud, line, made, star = namespace["figure"].data
    assert len(cloud.x) + len(guessed.x) == sum(meets)
    assert len(guessed.x) == sum(met and f for met, f in zip(meets, far, strict=True))
    assert len(made.x) == len(held["rows"]), "every run made has both results"
    np.testing.assert_allclose(line.x, np.sort(np.asarray(candidates["responses"]["Yield"])[front]), rtol=1e-5)
    assert (np.diff(line.x) >= 0).all(), "joined from left to right"
    by = {r["name"]: r["predicted"] for r in balance["responses"]}
    assert [star.x[0], star.y[0]] == pytest.approx([by["Yield"], by["Cost"]], abs=1e-6)


def assert_same_target_settings(namespace: dict, found: dict, spec: dict) -> None:
    """Assert that the code finds the page's settings for the targets, its predictions there, and its share."""
    ranges = {f["name"]: f["high"] - f["low"] for f in spec["factors"] if isinstance(f.get("low"), (int, float))}
    for setting in found["settings"]:
        if setting["kind"] == "numeric":
            tolerance = 1e-5 * ranges[setting["name"]]
            assert namespace["settings"][setting["name"]] == pytest.approx(setting["value"], abs=tolerance)
    for r in found["responses"]:
        assert namespace["reached"][r["name"]] == pytest.approx(r["predicted"], rel=1e-5), r["name"]
    if not found["search"]["held"]:  # the page's share is over every level it tried; the code's at the level found
        assert float(namespace["giving"].mean()) == pytest.approx(found["share"], abs=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("targets", "reached"),
    [({"Yield": 62.0}, True), ({"Yield": 62.0, "Cost": 60.0}, True), ({"Yield": 500.0}, False)],
    ids=["one target", "two targets", "out of reach"],
)
def test_the_target_settings_code_finds_the_page_s_settings(api, node, targets, reached):
    held, simulation = three_responses(api)
    request = {**held, **sent_of(by_name(simulation)), "models": {"Cost": "interactions"}}
    found = api.ok("api_target_settings", {**request, "targets": targets})
    assert found["reached"] is reached
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, by_name(simulation), simulation]),
        ("targetCode", [found, held["spec"]]),
    )
    assert_same_target_settings(run(code), found, held["spec"])


@pytest.mark.slow
@pytest.mark.parametrize(
    "targets",
    [{"Hardness": None}, {"Hardness": None, "Cost": None}, {"Hardness": 1e6}],
    ids=["one target", "two targets", "out of reach"],
)
def test_the_target_code_finds_the_page_s_blend(api, node, targets):
    """A mixture's code draws the page's blends from the region of blends, and steps to the targets inside it."""
    responses = [{"name": "Hardness", "goal": "maximize"}, {"name": "Cost", "goal": "minimize"}]
    request = {**MIXTURES[1], "factors": components("simplex_lattice"), "responses": responses}
    held = held_of(api.ok("api_make_design", request))
    simulation = api.ok("api_simulate", {**held, "seed": 5})
    results = {"Hardness": simulation["responses"], "Cost": simulation["more"]["Cost"]["responses"]}
    # A target left open here is the middle of the response's results, which some blend gives.
    middle = {name: (min(values) + max(values)) / 2 for name, values in results.items()}
    asked = {name: middle[name] if target is None else target for name, target in targets.items()}
    sent = {"responses": results["Hardness"], "more_responses": {"Cost": results["Cost"]}}
    found = api.ok("api_target_settings", {**held, **sent, "models": {"Cost": "scheffe_linear"}, "targets": asked})
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results, simulation]),
        ("targetCode", [found, held["spec"]]),
    )
    assert "DesignRegion(factors).sample(" in code
    namespace = run(code)
    assert sum(namespace["settings"].values()) == pytest.approx(1.0)
    for name, share in found["coded"].items():
        assert namespace["settings"][name] == pytest.approx(share, abs=1e-5), name
    for r in found["responses"]:
        assert namespace["reached"][r["name"]] == pytest.approx(r["predicted"], rel=1e-5), r["name"]
    assert float(namespace["giving"].mean()) == pytest.approx(found["share"], abs=1e-12)


@pytest.mark.slow
def test_the_target_settings_code_holds_a_categorical_factor_and_keeps_to_the_limits(api, node):
    factors = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]
    spec = {"design_type": "full_factorial", "factors": factors, "responses": THREE[:2], "n_center_points": 3}
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), factors)
    yields = (60 + 4 * x["T"] - 2 * x["P"] + 3 * x["S"] + x["T"] * x["P"]).tolist()
    costs = (20 + 2 * x["T"] + x["P"] + 0.5 * x["S"]).tolist()
    request = {**held, "responses": yields, "more_responses": {"Cost": costs}, "models": {"Yield": "interactions"}}
    found = api.ok("api_target_settings", {**request, "targets": {"Yield": 66.0}})  # reached only with S at B
    assert found["search"]["held"] == {"S": 1.0}
    steps = (("designCode", [held["spec"]]), ("resultsCode", [held, {"Yield": yields, "Cost": costs}]))
    code = panel(node, *steps, ("targetCode", [found, held["spec"]]))
    assert "held_at(" in code
    assert_same_target_settings(run(code), found, held["spec"])

    limit = [{"expression": "T + 25*P <= 215"}]
    held = held_of(
        api.ok(
            "api_make_design",
            {"design_type": "d_optimal", "factors": FACTORS[:2], "responses": THREE[:2], "constraints": limit},
        )
    )
    x = coded(run_sheet(held), FACTORS[:2])
    yields = (60 + 4 * x["T"] + 2 * x["P"] - 3 * x["T"] ** 2).tolist()
    costs = (20 + 3 * x["T"] + x["P"]).tolist()
    request = {**held, "responses": yields, "more_responses": {"Cost": costs}}
    found = api.ok("api_target_settings", {**request, "targets": {"Yield": 63.0, "Cost": 23.5}})
    steps = (("designCode", [held["spec"]]), ("resultsCode", [held, {"Yield": yields, "Cost": costs}]))
    code = panel(node, *steps, ("targetCode", [found, held["spec"]]))
    assert "region = DesignRegion(factors, constraints)" in code
    namespace = run(code)
    assert_same_target_settings(namespace, found, held["spec"])
    assert namespace["settings"]["T"] + 25 * namespace["settings"]["P"] <= 215 + 1e-6


@pytest.mark.slow
def test_the_target_settings_code_searches_a_transformed_response_on_its_scale(api, node):
    held, simulation = three_responses(api)
    held = {**held, "spec": {**held["spec"], "analysis_transforms": {"Yield": "log"}}}
    request = {**held, **sent_of(by_name(simulation))}
    found = api.ok("api_target_settings", {**request, "targets": {"Yield": 62.0, "Cost": 60.0}})
    assert found["search"]["targets"]["Yield"] == pytest.approx(np.log(62.0))
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, by_name(simulation), simulation]),
        ("targetCode", [found, held["spec"]]),
    )
    assert_same_target_settings(run(code), found, held["spec"])


@pytest.mark.slow
@pytest.mark.parametrize("transform", [None, "log"])
def test_the_target_chart_s_code_draws_each_response_s_line_as_the_page_does(api, node, monkeypatch, transform):
    held, simulation = three_responses(api)
    if transform:
        held = {**held, "spec": {**held["spec"], "analysis_transforms": {"Yield": transform}}}
    request = {**held, **sent_of(by_name(simulation)), "models": {"Cost": "interactions"}}
    found = api.ok("api_target_settings", {**request, "targets": {"Yield": 62.0, "Cost": 60.0}, "pair": ["T", "F"]})
    overlay = found["overlay"]
    assert [overlay["x"], overlay["y"]] == ["T", "F"], "the two factors the reader chose"
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, by_name(simulation), simulation]),
        ("targetCode", [found, held["spec"]]),
        ("targetOverlayCode", [overlay, held["spec"]]),
    )
    yields, costs, star = drawn_by(monkeypatch, code)["figure"].data
    for line, surface, target in zip((yields, costs), overlay["responses"], (62.0, 60.0), strict=True):
        assert line.name == surface["name"]
        np.testing.assert_allclose(np.asarray(line.z, dtype=float), surface["z"], rtol=1e-5)  # the page rounds to six
        assert line.contours.start == line.contours.end == pytest.approx(target)
        assert surface["levels"] == pytest.approx([target])
    assert [star.x[0], star.y[0]] == pytest.approx(overlay["at"], abs=1e-5)


#: The settings the specification's tests judge through: the center of the worked example's ranges.
CENTER = {"T": 175, "P": 2, "F": 15}


@pytest.mark.slow
@pytest.mark.parametrize("windowed", [("Hardness",), ("Hardness", "Cost")], ids=["one window", "two windows"])
def test_the_specification_code_judges_the_page_s_blends(api, node, windowed):
    """A mixture's code draws the page's blends from the region of blends and judges them as the page does."""
    responses = [{"name": "Hardness", "goal": "maximize"}, {"name": "Cost", "goal": "minimize"}]
    request = {**MIXTURES[1], "factors": components("simplex_lattice"), "responses": responses}
    held = held_of(api.ok("api_make_design", request))
    simulation = api.ok("api_simulate", {**held, "seed": 5})
    results = {"Hardness": simulation["responses"], "Cost": simulation["more"]["Cost"]["responses"]}
    # Each window is the upper half of the response's results, an end left open for the second.
    low = {name: (min(values) + max(values)) / 2 for name, values in results.items()}
    windows = {name: [low[name], None] for name in windowed}
    sent = {"responses": results["Hardness"], "more_responses": {"Cost": results["Cost"]}}
    found = api.ok("api_specification", {**held, **sent, "models": {"Cost": "scheffe_linear"}, "windows": windows})
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, results, simulation]),
        ("windowCode", [found, held["spec"]]),
    )
    assert "DesignRegion(factors).sample(" in code
    namespace = run(code)
    assert_same_specification(namespace, found)
    if found["likely"] is not None:
        assert sum(namespace["settings"].values()) == pytest.approx(1.0)


def assert_same_specification(namespace: dict, found: dict) -> None:
    """Assert that the code judges the page's settings: the same share, the same chances, the same surest settings."""
    judged = namespace["judged"]
    assert len(judged) == found["tried"]
    assert float(judged["feasible"].mean()) == pytest.approx(found["share"], abs=1e-12)
    if found["likely"] is None:
        assert judged["probability"].isna().all()
        return
    for chance, share in found["likely"].items():
        assert float(namespace["likely"][float(chance)]) == pytest.approx(share, abs=1e-12), chance
    surest = found["surest"]
    assert float(judged["probability"][namespace["surest"]]) == pytest.approx(surest["probability"], abs=1e-9)
    for name, value in namespace["settings"].items():
        assert value == pytest.approx(surest["settings"][name], rel=1e-9), name


@pytest.mark.slow
@pytest.mark.parametrize("transform", [None, "log"])
def test_the_specification_s_code_judges_the_page_s_settings_and_draws_its_chart(api, node, monkeypatch, transform):
    held, simulation = three_responses(api)
    if transform:
        held = {**held, "spec": {**held["spec"], "analysis_transforms": {"Yield": transform}}}
    request = {**held, **sent_of(by_name(simulation)), "models": {"Cost": "interactions"}}
    windows = {"Yield": [58.0, None], "Cost": [None, 62.0]}
    found = api.ok("api_specification", {**request, "windows": windows, "at": {**CENTER, "P": 2.5}, "pair": ["T", "F"]})
    assert 0 < found["likely"]["0.9"] < found["share"] < 1
    overlay = found["overlay"]
    correlation = found["search"]["correlation"]
    assert correlation is not None, "the runs leave enough to take the responses' errors as correlated"
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, by_name(simulation), simulation]),
        ("windowCode", [found, held["spec"]]),
        ("windowOverlayCode", [overlay, held["spec"], found["search"]["seed"]]),
    )
    namespace = drawn_by(monkeypatch, code)
    assert_same_specification(namespace, found)
    np.testing.assert_allclose(namespace["correlation"], correlation, atol=1e-9)
    assert namespace["correlation_df"] == found["search"]["correlation_df"] >= 5, (
        "and the doubt in it, as the page has it"
    )
    shaded, chance, star = namespace["figure"].data
    np.testing.assert_allclose(np.asarray(chance.z, dtype=float), overlay["probability"], atol=2e-6)
    assert (chance.contours.start, chance.contours.end) == (0.5, 0.9)
    assert [star.x[0], star.y[0]] == pytest.approx(overlay["at"])
    # The page shades inside the window's ends and the code on them too: the two agree off the ends themselves.
    surfaces = [np.asarray(s["z"]) for s in overlay["responses"]]
    off = (np.abs(surfaces[0] - 58.0) > 1e-3) & (np.abs(surfaces[1] - 62.0) > 1e-3)
    assert np.array_equal(np.asarray(shaded.z, dtype=int)[off], np.asarray(overlay["acceptable"])[off])


@pytest.mark.slow
def test_the_specification_s_code_holds_a_categorical_factor_and_keeps_to_the_limits(api, node):
    factors = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]
    spec = {"design_type": "full_factorial", "factors": factors, "responses": THREE[:2], "n_center_points": 3}
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), factors)
    noise = np.random.default_rng(2).normal(0.0, 0.3, len(x))
    yields = (60 + 4 * x["T"] - 2 * x["P"] + 3 * x["S"] + x["T"] * x["P"] + noise).tolist()
    costs = (20 + 2 * x["T"] + x["P"] + 0.5 * x["S"] + noise[::-1]).tolist()
    request = {**held, "responses": yields, "more_responses": {"Cost": costs}, "models": {"Yield": "interactions"}}
    asked = {"windows": {"Yield": [62.0, None], "Cost": [None, 22.0]}, "at": {"T": 175, "P": 2, "S": "B"}}
    found = api.ok("api_specification", {**request, **asked})
    assert found["search"]["at"]["S"] == 1.0
    steps = (("designCode", [held["spec"]]), ("resultsCode", [held, {"Yield": yields, "Cost": costs}]))
    assert_same_specification(run(panel(node, *steps, ("windowCode", [found, held["spec"]]))), found)

    limit = [{"expression": "T + 25*P <= 215"}]
    held = held_of(
        api.ok(
            "api_make_design",
            {"design_type": "d_optimal", "factors": FACTORS[:2], "responses": THREE[:2], "constraints": limit},
        )
    )
    x = coded(run_sheet(held), FACTORS[:2])
    noise = np.random.default_rng(4).normal(0.0, 0.2, len(x))
    yields = (60 + 4 * x["T"] + 2 * x["P"] - 3 * x["T"] ** 2 + noise).tolist()
    costs = (20 + 3 * x["T"] + x["P"] + noise[::-1]).tolist()
    request = {**held, "responses": yields, "more_responses": {"Cost": costs}}
    found = api.ok("api_specification", {**request, "windows": {"Cost": [None, 19.0]}, "at": {"T": 160, "P": 1.5}})
    steps = (("designCode", [held["spec"]]), ("resultsCode", [held, {"Yield": yields, "Cost": costs}]))
    code = panel(node, *steps, ("windowCode", [found, held["spec"]]))
    assert "region = DesignRegion(factors, constraints)" in code
    namespace = run(code)
    assert_same_specification(namespace, found)
    assert namespace["settings"]["T"] + 25 * namespace["settings"]["P"] <= 215 + 1e-6


@pytest.mark.slow
def test_the_specification_s_code_fits_a_study_in_blocks_at_the_average_block(api, node):
    spec = {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 2, "n_blocks": 2}
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), FACTORS)
    block = np.where(np.asarray([row[held["columns"].index("Block")] for row in held["rows"]]) == 1, 1.5, -1.5)
    yields = (60 + 4 * x["T"] - 2 * x["P"] + block + np.random.default_rng(6).normal(0.0, 0.3, len(x))).tolist()
    name = held["spec"]["response"]
    found = api.ok("api_specification", {**held, "responses": yields, "windows": {name: [59.0, 63.0]}, "at": CENTER})
    assert found["search"]["at"]["Block1"] == 0.0
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, yields]),
        ("windowCode", [found, held["spec"]]),
    )
    assert 'columns = X.drop(columns="Block").assign(**contrasts)' in code
    assert_same_specification(run(code), found)


@pytest.mark.slow
def test_the_specification_s_code_gives_no_chance_when_the_fit_leaves_no_run_for_the_noise(api, node, monkeypatch):
    spec = {"design_type": "full_factorial", "factors": FACTORS[:2], "responses": THREE[:1], "n_center_points": 0}
    held = held_of(api.ok("api_make_design", spec))
    x = coded(run_sheet(held), FACTORS[:2])
    yields = (60 + 4 * x["T"] - 2 * x["P"] + x["T"] * x["P"]).tolist()
    request = {**held, "responses": yields, "models": {"Yield": "interactions"}}
    found = api.ok("api_specification", {**request, "windows": {"Yield": [59.0, 62.0]}, "at": {"T": 175, "P": 2}})
    assert found["likely"] is None
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, yields]),
        ("windowCode", [found, held["spec"]]),
        ("windowOverlayCode", [found["overlay"], held["spec"]]),
    )
    namespace = drawn_by(monkeypatch, code)
    assert_same_specification(namespace, found)
    assert [trace.name for trace in namespace["figure"].data] == [
        "Every response inside its window",
        "What-if settings",
    ]


@pytest.mark.slow
def test_the_target_settings_code_fits_a_factor_nested_in_another_as_the_page_does(api, node):
    """S + P + S:T, T's effect within each level of S: the code adds S's indicators to X, as the page fits them."""
    levels = {"name": "S", "type": "categorical", "levels": ["a", "b", "c"], "units": ""}
    spec = {"design_type": "full_factorial", "factors": [*FACTORS[:2], levels], "n_center_points": 0, "n_replicates": 2}
    held = held_of(api.ok("api_make_design", spec))
    sheet = run_sheet(held)
    x = coded(sheet, FACTORS[:2])
    slope = sheet["S"].map({"a": 1.0, "b": 3.0, "c": 5.0})
    responses = (60 + slope * x["T"] + 2 * x["P"] + np.arange(len(sheet)) % 4 * 0.1).tolist()
    name = held["spec"]["response"]
    request = {**held, "responses": responses, "models": {name: {"terms": [["S"], ["P"], ["S", "T"]]}}}
    found = api.ok("api_target_settings", {**request, "targets": {name: 64.0}})
    assert found["reached"]
    code = panel(
        node,
        ("designCode", [held["spec"]]),
        ("resultsCode", [held, responses]),
        ("targetCode", [found, held["spec"]]),
    )
    assert 'X["S_I1"] = (runs["S"] == "a").astype(float)' in code
    assert_same_target_settings(run(code), found, held["spec"])
