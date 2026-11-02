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
import io
import json

import numpy as np
import pytest

from tests.web._support import (
    ALL_DESIGNS,
    FACTORS,
    coded,
    correlated_study,
    generate_all,
    held_of,
    javascript,
    no_bytecode,
    run_sheet,
    source,
)
from tests.web.api.test_book_examples import catalogue


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
