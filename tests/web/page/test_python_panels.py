"""The "Show the Python" panels: the code each step of the page shows must do what the page did.

``web/python.js`` writes the code from the page's state. It is run here under Node, as the
page runs it, and the code it writes is run in Python, so the code a reader copies is checked
to reproduce what the page computed.
"""

from __future__ import annotations

import contextlib
import io

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

    run(page_code(node, "simulateCode", simulation), namespace)
    np.testing.assert_allclose(namespace["y"], simulation["responses"], atol=0.005, err_msg="the page rounds")

    run(page_code(node, "analyseCode", analysis["spec"], analysis["model"]), namespace)
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
