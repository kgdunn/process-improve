"""The "Show the Python" panels: the code each step of the page shows must do what the page did.

``web/python.js`` writes the code from the page's state. It is run here under Node, as the
page runs it, and the code it writes is run in Python, so the code a reader copies is checked
to reproduce what the page computed.
"""

from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pytest

from tests.web._support import ALL_DESIGNS, FACTORS, generate_all, held_of, javascript, run_sheet, source


def page_code(node: str, function: str, *args: object) -> str:
    """Return the code one of ``web/python.js``'s writers gives for the page's state."""
    return javascript(node, source("python.js"), [(function, list(args))])[0]


def run(code: str, namespace: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)  # noqa: S102 - the page's own code, the subject of this test
    return namespace


@pytest.mark.slow
def test_the_python_shown_reproduces_the_page(api, node):
    demo = api.ok("api_catalogue")["demo"]
    design = api.ok("api_make_design", demo["design"])
    held = held_of(design)
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 1})
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulation["responses"]})

    namespace = run(page_code(node, "designCode", design["spec"]), {})
    names = [f["name"] for f in design["spec"]["factors"]]
    page_runs = run_sheet(design)[names]
    np.testing.assert_allclose(namespace["result"].design_actual[names].to_numpy(dtype=float), page_runs, atol=1e-9)

    run(page_code(node, "simulateCode", simulation, design["spec"]), namespace)
    np.testing.assert_allclose(namespace["y"], simulation["responses"], atol=0.005, err_msg="the page rounds")

    run(page_code(node, "analyseCode", analysis["spec"], analysis["formula"]), namespace)
    shown = {c["term"]: c["coefficient"] for c in namespace["analysis"]["coefficients"]}
    for c in analysis["coefficients"]:
        assert shown[c["term"]] == pytest.approx(c["coefficient"], abs=0.01), c["term"]

    entered = run(page_code(node, "enteredCode", [*simulation["responses"][:-1], None]), {})
    assert entered["y"].isna().tolist() == [False] * 14 + [True]

    made, held_designs = generate_all(api, FACTORS, "main_quadratic")
    compared = api.ok("api_compare", {"model": "main_quadratic", "designs": held_designs})
    designs = [
        {"label": m["label"], "design_type": m["design_type"], "options": m["design"]["spec"]["options"]} for m in made
    ]
    namespace = run(page_code(node, "compareCode", FACTORS, designs, "main_quadratic"), {})
    assert list(namespace["comparison"].designs) == compared["designs"]
    for label in compared["designs"]:
        assert namespace["comparison"].table[label].tolist() == compared["display"][label], label


@pytest.mark.slow
def test_the_design_code_reproduces_every_design_the_page_offers(api, node):
    """Each design type's options, its seed included, reach the code, so the copied code gives the same runs."""
    designs = [api.ok("api_make_design", {"design_type": d, "factors": FACTORS}) for d in ALL_DESIGNS]
    codes = javascript(node, source("python.js"), [("designCode", [d["spec"]]) for d in designs])
    names = [f["name"] for f in FACTORS]
    for design_type, design, code in zip(ALL_DESIGNS, designs, codes, strict=True):
        result = run(code, {})["result"]
        np.testing.assert_allclose(
            result.design_actual[names].to_numpy(dtype=float), run_sheet(design)[names], err_msg=design_type
        )


@pytest.mark.parametrize("units", ['in. "wide"', "C:\\temp", "°C", "mol/L · s", ""])
def test_the_design_code_keeps_the_units_a_reader_types(api, node, units):
    """Units are free text: whatever the reader types, the code must still run and say the same."""
    spec = {"design_type": "full_factorial", "factors": [{**f, "units": units} for f in FACTORS[:2]]}
    design = api.ok("api_make_design", spec)
    factors = run(page_code(node, "designCode", design["spec"]), {})["factors"]
    assert [f.units for f in factors] == [units, units]


def test_results_typed_in_are_listed_in_run_order_with_none_for_runs_to_do(node):
    y = run(page_code(node, "enteredCode", [61.25, None, -3, 1e-7]), {})["y"]
    assert y.name == "y"
    np.testing.assert_array_equal(y.to_numpy(dtype=float), [61.25, np.nan, -3, 1e-7])


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
    ],
    ids=[
        "blocks confounding an interaction",
        "typed axial distance, replicated",
        "blocks by exchange",
        "a categorical factor, in blocks",
    ],
)
def test_the_python_shown_reproduces_blocks_replicates_and_a_typed_axial_distance(api, node, spec):
    """The design, its simulated results and their analysis, as the page shows the code for them."""
    design = api.ok("api_make_design", spec)
    held = held_of(design)
    simulation = api.ok("api_simulate", {**held, "scenario": "reactor", "sigma": 1.0, "seed": 3})
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulation["responses"]})

    namespace = run(page_code(node, "designCode", design["spec"]), {})
    columns = [c for c in design["columns"] if c in namespace["result"].design_actual.columns or c == "Block"]
    assert namespace["result"].design_actual[columns].astype(str).to_numpy().tolist() == (
        run_sheet(design)[columns].astype(str).to_numpy().tolist()
    )
    run(page_code(node, "simulateCode", simulation, design["spec"]), namespace)
    run(page_code(node, "analyseCode", analysis["spec"], analysis["formula"]), namespace)
    shown = {c["term"]: c["coefficient"] for c in namespace["analysis"]["coefficients"]}
    assert set(shown) == {c["term"] for c in analysis["coefficients"]}
    for c in analysis["coefficients"]:
        assert shown[c["term"]] == pytest.approx(c["coefficient"], abs=0.01), c["term"]


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
        spec=design["spec"],
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
        page_code(node, "simulateCode", simulation, design["spec"]),
        page_code(node, "analyseCode", design["spec"], analysis["formula"]),
    ]
    namespace: dict = {}
    for cell in code[1:]:
        run(cell, namespace)
    shown = {c["term"]: c["coefficient"] for c in namespace["analysis"]["coefficients"]}
    for c in analysis["coefficients"]:
        assert shown[c["term"]] == pytest.approx(c["coefficient"], abs=0.01), c["term"]


def test_a_notebook_of_typed_results_writes_them_out_and_waits_for_an_analysis(api, node):
    """Results typed in, or a simulation edited, are written out; with no analysis yet there is no analysis cell."""
    design = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 0})
    responses = [1.5, None, 2.5, 3.0, None, 4.0, 4.5, 5.0]
    book = notebook_of(node, spec=design["spec"], responses=responses, sources=["entered"] * 8, model=None)
    code = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
    assert code[-1] == page_code(node, "enteredCode", responses)
    assert not any("analyze_experiment" in cell for cell in code)
    unpinned = notebook_of(
        node, spec=design["spec"], responses=responses, sources=["entered"] * 8, model=None, version=None
    )
    assert "".join(unpinned["cells"][1]["source"]) == "%pip install process-improve"
