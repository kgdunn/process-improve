"""Every ``@tool_spec`` tool gives the same output when it is called twice with the same input.

``docs/development/reproducibility.rst`` sets the contract: a tool that declares
``rng={"uses_rng": True, ...}`` reproduces its output from its ``default_seed``, given through its
``seed_param``, and every other tool reproduces its output with no seed. ``@tool_spec`` checks only
the declaration; these tests run each tool twice through ``execute_tool_call``, as an agent calls it,
and compare the JSON of the two outputs byte for byte.

``SAMPLE_INPUTS`` holds one valid input per registered tool, and ``NOT_REPRODUCIBLE`` names each tool
that by design does not give the same output twice, with the reason. A registered tool in neither
table fails ``test_every_registered_tool_has_a_sample_input``, so a new tool cannot skip the check. A tool
whose input takes a seed that its ``rng=`` does not declare fails ``test_every_tool_that_takes_a_seed_declares_it``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, TrueProcess, generate_design
from process_improve.simulation.context import simulator_host_context
from process_improve.tool_spec import _TOOL_REGISTRY, clean, discover_tools, execute_tool_call, get_tool_specs

_RNG = np.random.default_rng(0)
_VALUES = np.round(_RNG.normal(10.0, 1.0, 20), 3).tolist()
_OTHER = np.round(_RNG.normal(10.5, 1.0, 20), 3).tolist()
_LINE = [2.0 * i + v - 10.0 for i, v in enumerate(_VALUES)]
_MATRIX = np.round(_RNG.normal(size=(15, 4)), 3).tolist()
_X = [row[:3] for row in _MATRIX]
_Y = [[round(row[0] - 0.5 * row[1] + 0.1 * row[3], 3)] for row in _MATRIX]
_TABLE = pd.DataFrame(_MATRIX, columns=["a", "b", "c", "y"], index=pd.Index([f"r{i}" for i in range(15)], name="id"))

_FACTORS = [{"name": "A", "low": 0, "high": 10}, {"name": "B", "low": 0, "high": 10}]
_FACTORIAL = [
    {"A": -1, "B": -1, "y": 28.0},
    {"A": 1, "B": -1, "y": 36.0},
    {"A": -1, "B": 1, "y": 18.0},
    {"A": 1, "B": 1, "y": 31.0},
    {"A": -1, "B": -1, "y": 27.5},
    {"A": 1, "B": -1, "y": 36.5},
    {"A": -1, "B": 1, "y": 17.5},
    {"A": 1, "B": 1, "y": 31.5},
]
_THREE_LEVELS = [{"A": a, "B": b} for a in (-1, 0, 1) for b in (-1, 0, 1)]
_HALF_CUBE = [{"A": -1, "B": -1, "C": c} for c in (-1, 1)] + [{"A": -1, "B": 1, "C": c} for c in (-1, 1)]
_QUADRATIC = [
    {"term": "Intercept", "coefficient": 40.0},
    {"term": "A", "coefficient": 5.25},
    {"term": "B", "coefficient": -2.0},
    {"term": "I(A ** 2)", "coefficient": -3.0},
    {"term": "I(B ** 2)", "coefficient": -1.5},
    {"term": "A:B", "coefficient": 1.5},
]

_PANEL = [
    {
        "panelist_id": f"P{p}",
        "session": 1,
        "product": f"prod{q}",
        "attribute": "A",
        "replicate": r,
        "score": round(5 + q + 0.3 * p + float(_RNG.normal(0, 0.4)), 3),
    }
    for p in range(6)
    for q in range(5)
    for r in (1, 2)
]
# Two descriptors: one alone has a selectivity ratio of 0 in every permutation, so its p-value would not use the seed.
_PRODUCT_COVARIATES = [{"product": f"prod{q}", "d": float(q), "e": float(3 * q % 5)} for q in range(5)]
# One factor at a time leaves A:B unestimable, so recommend_next_step chooses the next runs by exchange, from its seed.
_ONE_AT_A_TIME = [
    {"A": 5.0 + 5 * a, "B": 5.0 + 5 * b, "y": round(40 + 5 * a + 3 * b + v, 3)}
    for (a, b), v in zip([(0, 0)] * 3 + [(-1, 0), (1, 0), (0, -1), (0, 1)] * 2, _VALUES, strict=False)
]

_SIMULATOR = {
    "process_description": "A small reactor",
    "factors": [{"name": "T", "low": 150, "high": 200}, {"name": "P", "low": 1, "high": 5}],
    "outputs": [{"name": "yield", "units": "%", "direction": "maximize"}],
}


def _dsd_runs() -> list[dict[str, Any]]:
    """Return a six-factor DSD with three centre runs, coded, and a response in which A, C and A:C are active."""
    design = generate_design([Factor(name=n, low=-1, high=1) for n in "ABCDEF"], design_type="dsd", n_center_points=3)
    runs = design.design.drop(columns="RunOrder")
    truth = TrueProcess({"Intercept": 50, "A": 6, "C": -4, "A:C": 4}, sigma=0.5)
    return runs.assign(y=truth.simulate(runs, random_state=5)).to_dict("records")


def _session(*, fitted: bool) -> dict[str, Any]:
    """Return a multivariate session on ``_TABLE``, holding one PCA model ``M1`` when ``fitted``."""
    session = execute_tool_call("mva_open", {"source": {"csv": _TABLE.to_csv(), "name": "plant.csv"}})["session"]
    if fitted:
        request = {"session": session, "name": "M1", "kind": "PCA", "n_components": 2, "cross_validate": False}
        session = execute_tool_call("mva_fit", request)["session"]
    return session


#: One valid input per registered tool. A function builds an input that holds another tool's output.
#: A tool that uses an RNG gets its seed added by the test, so its input leaves the seed out.
SAMPLE_INPUTS: dict[str, dict[str, Any] | Callable[[], dict[str, Any]]] = {
    # univariate
    "robust_summary_stats": {"values": _VALUES},
    "detect_outliers": {"values": [*_VALUES, 25.0]},
    "robust_scale_sn": {"values": _VALUES},
    "median_absolute_deviation": {"values": _VALUES},
    "normality_test": {"values": _VALUES},
    "confidence_interval": {"values": _VALUES},
    "ttest_two_samples": {"group_a": _VALUES, "group_b": _OTHER},
    "ttest_paired_samples": {"group_a": _VALUES, "group_b": _OTHER},
    "within_between_variance": {"values": _VALUES, "groups": ["a"] * 10 + ["b"] * 10},
    # multivariate
    "fit_pca": {"data": _MATRIX, "n_components": 2},
    "fit_pls": {"x_data": _X, "y_data": _Y, "n_components": 2},
    "scale_data": {"data": _MATRIX},
    "detect_multivariate_outliers": {"data": _MATRIX, "n_components": 2},
    "pca_predict": lambda: {
        "model_params": execute_tool_call("fit_pca", {"data": _MATRIX, "n_components": 2})["model_params"],
        "new_data": _MATRIX[:3],
    },
    "pls_predict": lambda: {
        "model_params": execute_tool_call("fit_pls", {"x_data": _X, "y_data": _Y, "n_components": 2})["model_params"],
        "new_data": _X[:3],
    },
    "mva_open": {"source": {"csv": _TABLE.to_csv(), "name": "plant.csv"}},
    # cross_validate is left at its default, True: the fold assignment is the step that draws random numbers.
    "mva_fit": lambda: {"session": _session(fitted=False), "name": "M1", "kind": "PCA", "n_components": 2},
    "mva_explain": lambda: {"session": _session(fitted=True), "model": "M1", "rows": ["r3"]},
    "mva_predict": lambda: {"session": _session(fitted=True), "model": "M1", "rows": ["r0", "r1"]},
    "mva_script": lambda: {"session": _session(fitted=True)},
    # monitoring, regression, bivariate, batch, visualization
    "control_chart": {"values": _VALUES},
    "process_capability": {"values": _VALUES, "lower_spec": 7.0, "upper_spec": 13.0},
    "robust_regression": {"x": list(range(20)), "y": _LINE},
    "repeated_median": {"x": list(range(20)), "y": _LINE},
    "find_elbow": {"x": list(range(20)), "y": [min(i, 8) + 0.01 * v for i, v in enumerate(_VALUES)]},
    "extract_batch_features": {
        "data": [
            {"batch": f"B{b}", "time": t, "temp": 100 + b + t + v} for b in range(3) for t, v in enumerate(_VALUES)
        ],
        "value_columns": ["temp"],
        "features": ["mean", "std", "slope"],
        "time_column": "time",
    },
    "boxplot": {
        "data": [{"w": v, "lot": "AB"[i % 2]} for i, v in enumerate(_VALUES)],
        "value_columns": ["w"],
        "group_by": "lot",
    },
    # experiments
    "create_factorial_design": {"n_factors": 3},
    "fit_linear_model": {"formula": "y ~ A*B", "data": _FACTORIAL},
    "generate_design": {
        "factors": [*_FACTORS, {"name": "C", "low": 0, "high": 1}],
        "design_type": "ccd",
        "alpha": "face_centered",
    },
    "evaluate_design": {
        "design_matrix": _THREE_LEVELS,
        "metric": ["d_efficiency", "average_prediction_variance"],
        "model": "quadratic",
    },
    "analyze_experiment": {"design_matrix": _FACTORIAL, "response_column": "y", "model": "y ~ A*B"},
    "optimize_responses": {
        "fitted_models": [
            {"response_name": "y", "factor_names": ["A", "B"], "coefficients": _QUADRATIC},
            {"response_name": "z", "factor_names": ["A", "B"], "coefficients": _QUADRATIC},
        ],
        "goals": [
            {"response": "y", "goal": "maximize", "low": 30.0, "high": 45.0},
            {"response": "z", "goal": "minimize", "low": 20.0, "high": 45.0},
        ],
        "method": "pareto_front",
        "n_pareto_points": 5,
    },
    "augment_design": {
        "existing_design": _HALF_CUBE,
        "augmentation_type": "add_runs_optimal",
        "n_additional_runs": 3,
        "target_model": "interactions",
    },
    "visualize_doe": {"plot_type": "main_effects", "design_data": _FACTORIAL, "response_column": "y"},
    "doe_knowledge": {"query": "what is a fractional factorial design"},
    "recommend_strategy": {"factors": _FACTORS, "budget": 12},
    "trade_off_table": {"runs": [8, 16], "factors": [4, 5]},
    "search_design_catalog": {"families": ["box_behnken"], "max_factors": 3, "limit": 2},
    "analyze_omars": lambda: {"design_matrix": _dsd_runs(), "response_column": "y"},
    "recommend_next_step": {"factors": _FACTORS, "runs": _ONE_AT_A_TIME, "response_column": "y"},
    "write_run_sheet": {"runs": [{"A": r["A"], "B": r["B"]} for r in _FACTORIAL], "factors": _FACTORS},
    "read_run_sheet": {
        "csv_text": "Run,A,B,y\n1,0,0,28.0\n2,10,0,36.0\n3,0,10,18.0\n4,10,10,31.0\n",
        "factors": _FACTORS,
    },
    "simulate_design_performance": {
        "designs": {"3^2": _THREE_LEVELS},
        "true_coefficients": {"Intercept": 10.0, "A": 2.0, "B": -1.0},
        "sigma": 1.0,
        "n_simulations": 5,
    },
    "compare_designs": {"designs": {"3^2": _THREE_LEVELS, "2^2": _THREE_LEVELS[::2]}, "model": "main_effects"},
    # simulation: reveal_simulator reads the hidden model from the host context the test sets up
    "reveal_simulator": {"sim_id": "sim"},
    "simulate_batch_campaign": {"n_batches": 5},
    "decompose_batch_quality_variance": {"n_batches": 10},
    "correct_batch_midcourse": {"n_batches": 1, "n_train": 30},
    "evaluate_batch_control_policy": {"n_train": 30, "n_test": 5},
    # recipes and sensory
    "select_analysis_recipe": {"query": "what drives liking from the measurements"},
    "sensory_reshape_to_long": {
        "data": _PANEL,
        "layout": "long",
        "panelist_id": "panelist_id",
        "product": "product",
        "attribute": "attribute",
        "score": "score",
        "replicate": "replicate",
    },
    "sensory_validate_descriptive": {"panel": _PANEL, "covariates": _PRODUCT_COVARIATES, "mode": "observational"},
    "sensory_analyze_descriptive": {"panel": _PANEL, "covariates": _PRODUCT_COVARIATES, "mode": "observational"},
    "sensory_panel_check": {"panel": _PANEL, "align": True},
}

#: Tools that by design give a different output on each call, and why.
NOT_REPRODUCIBLE: dict[str, str] = {
    "create_simulator": (
        "mints a new sim_id (uuid4) and created_at timestamp on every call; "
        "test_the_simulator_seed_fixes_all_but_the_id_and_the_time checks the rest"
    ),
    "simulate_process": "adds fresh noise on every call, the simulator's documented contract (reproducibility.rst, 4)",
}

#: Cases slower than the unit tier allows, and the one that builds the design catalog from its bundled data.
_MARKS = {
    "search_design_catalog": (pytest.mark.slow, pytest.mark.dataset),
    "sensory_analyze_descriptive": (pytest.mark.slow,),
}


def _production_tools() -> set[str]:
    """Return the names of the tools the package registers, leaving out those that test modules register."""
    discover_tools()
    return {name for name, func in _TOOL_REGISTRY.items() if func.__module__.startswith("process_improve.")}


def _output(tool: str, tool_input: dict[str, Any], simulator_state: dict[str, Any]) -> str:
    """Call ``tool`` as an agent does and return its output as the JSON the agent receives."""
    with simulator_host_context(simulator_state=simulator_state, confirmed=True):
        return json.dumps(clean(execute_tool_call(tool, tool_input)))


@pytest.fixture(scope="module")
def simulator_state() -> dict[str, Any]:
    """Return a fixed hidden model, which the simulator tools read from the host rather than from their input."""
    return execute_tool_call("create_simulator", {**_SIMULATOR, "seed": 7})["_private"]


def test_every_registered_tool_has_a_sample_input() -> None:
    """A tool registered with neither a sample input nor a reason it cannot be reproduced fails here."""
    registered = _production_tools()
    assert set(SAMPLE_INPUTS).isdisjoint(NOT_REPRODUCIBLE)
    assert sorted(registered - SAMPLE_INPUTS.keys() - NOT_REPRODUCIBLE.keys()) == [], "add each to SAMPLE_INPUTS"
    assert sorted((SAMPLE_INPUTS.keys() | NOT_REPRODUCIBLE.keys()) - registered) == [], "no tool has these names"


def test_every_tool_that_takes_a_seed_declares_it() -> None:
    """A tool whose input has a seed field names it as ``seed_param``, with the field's default as ``default_seed``.

    The other way round too: a tool that declares ``uses_rng`` takes the seed it names, with the default it names.
    """
    declared, taken = {}, {}
    for tool in _production_tools():
        rng = get_tool_specs([tool])[0].get("rng") or {"uses_rng": False}
        if rng["uses_rng"]:
            declared[tool] = (rng.get("seed_param"), rng.get("default_seed"))
        fields = _TOOL_REGISTRY[tool]._input_model.model_fields
        for name in {"random_state", "seed"} & fields.keys():
            taken[tool] = (name, fields[name].default)
    assert declared == taken


@pytest.mark.parametrize("tool", [pytest.param(t, marks=_MARKS.get(t, ())) for t in sorted(SAMPLE_INPUTS)])
def test_the_tool_gives_the_same_output_twice(tool: str, simulator_state: dict[str, Any]) -> None:
    """Twice with the declared ``default_seed`` for a tool that uses an RNG, twice with no seed for any other."""
    entry = SAMPLE_INPUTS[tool]
    tool_input = entry() if callable(entry) else dict(entry)
    rng = get_tool_specs([tool])[0].get("rng") or {"uses_rng": False}
    if rng["uses_rng"]:
        tool_input[rng["seed_param"]] = rng["default_seed"]
    first = _output(tool, tool_input, simulator_state)
    result = json.loads(first)
    assert "error" not in result, f"the sample input of {tool} is not valid: {result['error']}"
    assert result.get("ok", True), f"the sample input of {tool} is not valid: {result.get('errors')}"
    assert _output(tool, tool_input, simulator_state) == first
    if rng["uses_rng"]:
        # Otherwise the seed may not reach the RNG, or the sample input may not draw from it at all.
        other_seed = {**tool_input, rng["seed_param"]: rng["default_seed"] + 1}
        assert _output(tool, other_seed, simulator_state) != first, f"{tool}'s output does not depend on its seed"


def test_the_simulator_seed_fixes_all_but_the_id_and_the_time() -> None:
    """``create_simulator`` gives the same output for the same seed but its ``sim_id`` and ``created_at``.

    It declares no ``default_seed``: with no seed it draws one, so the hidden model cannot be guessed.
    Another seed must give another hidden model, as ``reveal_simulator`` shows it.
    """
    seed_param = get_tool_specs(["create_simulator"])[0]["rng"]["seed_param"]

    def created(seed: int) -> tuple[str, str]:
        output = execute_tool_call("create_simulator", {**_SIMULATOR, seed_param: seed})
        del output["sim_id"], output["public"]["created_at"]
        revealed = json.loads(_output("reveal_simulator", {"sim_id": "sim"}, output["_private"]))
        return json.dumps(clean(output)), json.dumps(revealed["model"])

    first = created(7)
    assert created(7) == first
    assert created(8)[1] != first[1], "create_simulator's hidden model does not depend on its seed"
