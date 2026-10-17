"""The "Show the Python code" panels: the code each step of the page shows must do what the page did.

``web/python.js`` writes the code from the page's state. It is run here under Node, as the
page runs it, and the code it writes is run in Python, so the code a reader copies is checked
to reproduce what the page computed. A panel shows its step after the steps it builds on, and
each panel's code is run here on its own, in a fresh namespace, as a reader who copies only it
would run it.
"""

from __future__ import annotations

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
        ("analyseCode", [analysis["formula"], complete, vif_model]),
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
