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

from tests.web._support import FACTORS, generate_all, held_of, javascript, run_sheet, source


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
