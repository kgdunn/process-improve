"""The contract between the page's JavaScript and its Python: what each reply must hold.

``web/app.js``, ``web/limits.js`` (with ``web/corners.js`` and ``web/picture.js``), ``web/charts.js``
and ``web/python.js`` read the replies of ``web/bootstrap.py`` key by key. A key renamed on one side
and not the other breaks the page without breaking any Python, so ``READS`` lists every key the
JavaScript reads and the type it expects, and each entry point's reply is checked against it.
"""

from __future__ import annotations

import functools
import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._support import FACTORS, held_of, source

if TYPE_CHECKING:
    from collections.abc import Callable

#: What the page reads from each reply's result, and the type it expects there. ``a.b`` is a key
#: inside a key, ``a[]`` each item of a list, ``a{}`` each value of a dict, and a bare ``[]`` each
#: item of the lists reached so far. ``?`` allows null. A list may be empty unless it is followed
#: by ``[]``: then it must hold something, so the check reaches its items.
READS: dict[str, list[str]] = {
    "catalogue": [
        "demo.design.design_type: str",
        "demo.design.response: str",
        "demo.design.n_center_points: int",
        "demo.design.factors[].name: str",
        "demo.design.factors[].low: number",
        "demo.design.factors[].high: number",
        "demo.design.factors[].units: str",
        "demo.design.response_info.label: str",
        "demo.design.response_info.units: str",
        "demo.design.response_info.goal: str",
        "demo.design.response_info.typical: number",
        "demo.design.response_info.noise: number",
        "demo.design.response_info.smallest_change: number",
        "demo.design.aim: str",
        "demo.scenario: str",
        "demo.sigma: number",
        "demo.model: str",
        "demo.seed: int",
        "version: str?",
    ],
    "design": [
        "n_runs: int",
        "columns[]: str",
        "rows[]: list",
        "model: str",
        "outside_range: list",
        "facts.center_runs: int",
        "facts.notes[]: str",
        "spec.design_type: str",
        "spec.response: str",
        "spec.model: str",
        "spec.options: dict",
        "spec.factors[].name: str",
        "spec.factors[].low: number",
        "spec.factors[].high: number",
        "spec.factors[].units: str",
        "spec.response_info.label: str",
        "spec.response_info.units: str",
        "spec.aim: str",
    ],
    "simulation": [
        "scenario: str",
        "label: str",
        "sigma: number",
        "seed: int",
        "responses[]: number",
        "equation: str",
        "inactive: list",
        "truth[].term: str",
        "truth[].coefficient: number",
    ],
    "analysis": [
        "spec.factors[].name: str",
        "model: str",
        "formula: str",
        "equation: str",
        "n_completed: int",
        "pending_runs: list",
        "notes: list",
        "summary.r_squared: number?",
        "summary.r_squared_adj: number?",
        "summary.r_squared_pred: number?",
        "summary.df_residual: int",
        "coefficients[].term: str",
        "coefficients[].label: str",
        "coefficients[].coefficient: number",
        "coefficients[].std_error: number?",
        "coefficients[].ci_low: number?",
        "coefficients[].ci_high: number?",
        "coefficients[].p_value: number?",
        "coefficients[].true_value: number?",
        "anova[].source: str",
        "anova[].df: number",
        "anova[].sum_sq: number?",
        "anova[].F: number?",
        "anova[].p_value: number?",
        "lack_of_fit.f_statistic: number?",
        "lack_of_fit.p_value: number?",
        "lack_of_fit.df_pure_error: int",
    ],
    "saturated analysis": [
        "summary.r_squared_pred: number?",
        "lenth.PSE: number",
        "lenth.ME: number",
        "lenth.SME: number",
        "lenth.effects[].label: str",
        "lenth.effects[].effect: number",
        "lenth.effects[].active_ME: bool",
        "lenth.effects[].active_SME: bool",
    ],
    "charts": [
        "plotly_js: str",
        "contour_factors[]: str",
        "figures{}: str",
        "shaded: bool",
        "held: list",
        "held_best: bool",
    ],
    "charts through the best settings": [
        "held[].name: str",
        "held[].value: any",
        "held_best: bool",
    ],
    "best settings": [
        "goal: str",
        "target: number?",
        "flat: bool",
        "predicted: number",
        "reached: bool?",
        "level: number",
        "confidence[]: number",
        "prediction[]: number",
        "leverage: number",
        "max_leverage: number",
        "extrapolating: bool",
        "settings[].name: str",
        "settings[].kind: str",
        "settings[].value: any",
        "settings[].coded: number",
        "settings[].in_model: bool",
        "settings[].end: str?",
        "settings[].beyond: bool",
        "coded{}: number",
        "limits: list",
        "stationary.factors[]: str",
        "stationary.inside: bool",
        "stationary.kind: str",
        "ascent{}: number",
        "unclear: bool",
        "sensitivity[].name: str",
        "sensitivity[].step: number",
        "sensitivity[].down: number",
        "sensitivity[].up: number",
        "sensitivity[].down_runnable: bool",
        "sensitivity[].up_runnable: bool",
        "sensitivity[].in_model: bool",
        "confirm_at{}: any",
        "confirmation: list",
        "search.held: dict",
        "search.goal.goal: str",
        "search.goal.low: number",
        "search.goal.high: number",
    ],
    "best settings on a limit": [
        "limits[].index: int",
        "limits[].expression: str",
        "limits[].beyond: bool",
    ],
    "confirmation runs judged": [
        "confirmation[].rows[]: int",
        "confirmation[].at{}: any",
        "confirmation[].runs: int",
        "confirmation[].done: int",
        "confirmation[].mean: number?",
        "confirmation[].predicted: number",
        "confirmation[].interval[]: number",
        "confirmation[].inside: bool?",
    ],
    "best settings without a goal": [
        "goal: str",
    ],
    "design of several responses": [
        "columns[]: str",
        "spec.response: str",
        "spec.responses[].name: str",
        "spec.responses[].label: str",
        "spec.responses[].units: str",
        "spec.responses[].goal: str",
    ],
    "simulation of several responses": [
        "responses[]: number",
        "truth[].term: str",
        "more{}.responses[]: number",
        "more{}.truth[].term: str",
        "more{}.truth[].coefficient: number",
        "more{}.equation: str",
        "more{}.inactive: list",
        "more{}.sigma: number",
        "more{}.seed: int",
    ],
    "upload of several responses": [
        "spec.responses[].name: str",
        "spec.analysis_models: dict",
        "responses[]: number?",
        "sources[]: str?",
        "more_responses{}.[]: number?",
        "more_sources{}.[]: str?",
    ],
    "relabel of several responses": [
        "simulation.more{}.truth[].term: str",
        "simulation.more{}.truth[].coefficient: number",
        "simulation.more{}.equation: str",
        "simulation.more{}.inactive: list",
    ],
    "workbook": [
        "xlsx_base64: str",
    ],
    "relabel": [
        "spec.design_type: str",
        "spec.response: str",
        "spec.options: dict",
        "spec.factors[].name: str",
        "spec.factors[].low: number",
        "spec.factors[].high: number",
        "spec.factors[].units: str",
        "columns[]: str",
        "outside_range: list",
        "facts.center_runs: int",
        "facts.notes[]: str",
        "simulation.label: str",
        "simulation.sigma: number",
        "simulation.seed: int",
        "simulation.equation: str",
        "simulation.inactive: list",
        "simulation.truth[].term: str",
        "simulation.truth[].coefficient: number",
    ],
    "upload": [
        "spec.design_type: str",
        "spec.response: str",
        "spec.model: str",
        "spec.options: dict",
        "spec.factors[].name: str",
        "spec.factors[].low: number",
        "spec.factors[].high: number",
        "spec.factors[].units: str",
        "columns[]: str",
        "rows[]: list",
        "responses[]: number?",
        "sources[]: str?",
    ],
    "candidates": [
        "model: str",
        "notes: list",
        "waiting: list",
        "made[].label: str",
        "made[].design_type: str",
        "made[].warnings: list",
        "made[].design.n_runs: int",
        "made[].design.model: str",
        "made[].design.columns[]: str",
        "made[].design.rows[]: list",
        "made[].design.spec.response: str",
        "made[].design.spec.options: dict",
        "result.designs[]: str",
        "result.measures[].key: str",
        "result.measures[].group: str",
        "result.measures[].label: str",
        "result.measures[].description: str",
        "result.measures[].direction: str?",
        "result.display{}.[]: str",
        "result.notes{}: list",
        "result.terms[]: str",
        "result.effect_size: number",
        "result.significance_level: number",
        "result.g_points: any",
        "result.values{}.runs: int",
        "result.values{}.fits: bool",
        "result.values{}.residual_df: int",
        "result.values{}.power_main: number?",
        "result.values{}.power_interaction: number?",
        "result.values{}.power_quadratic: number?",
        "result.fds{}.fraction[]: number",
        "result.fds{}.variance[]: number",
        "result.correlations{}.terms[]: str",
        "result.correlations{}.matrix[].[]: number?",
    ],
    "slow candidates": [
        "waiting[].label: str",
        "waiting[].design_type: str",
    ],
    "design with limits": [
        "spec.constraints[].expression: str",
        "spec.constraints[].type: str",
        "facts.notes[]: str",
    ],
    "limits": [
        "constraints[].expression: str",
        "constraints[].note: str",
        "constraints[].type: str",
        "constraints[].removed: number",
        "constraints[].removes: list",
        "constraints[].redundant: str?",
        "constraints[].linear.lhs: str",
        "constraints[].linear.low: number?",
        "constraints[].linear.high: number?",
        "probes.center: int",
        "probes.columns[]: str",
        "probes.rows[].[]: number",
        "probes.ruled_out_by[]: list",
        "kept: number",
        "empty: bool",
        "center_allowed: bool",
        "bounds[].name: str",
        "bounds[].low: number?",
        "bounds[].high: number?",
        "refusal: str?",
    ],
    "limits with a problem": [
        "constraints[].expression: str",
        "constraints[].problem: str",
        "refusal: str",
    ],
    "limits proposed": [
        "proposals[].marked[]: int",
        "proposals[].expression: str",
        "proposals[].removes: list",
        "proposals[].removed: number",
        "proposals[].corner: int?",
        "proposals[].cuts[].name: str",
        "proposals[].cuts[].at: number",
        "proposals[].cuts[].cut: number?",
        "adjusted.marked: list",
        "adjusted.expression: str",
        "adjusted.removes: list",
        "adjusted.removed: number",
        "adjusted.cuts[].cut: number?",
    ],
    "limits not proposed": [
        "proposals[].marked: list",
        "proposals[].problem: str",
        "adjusted.problem: str",
    ],
    "picture": [
        "picture.pairs[].x: str",
        "picture.pairs[].y: str",
        "picture.pairs[].shadow.kind: str",
        "picture.pairs[].shadow.points: list",
        "picture.pairs[].lines[].limit: int",
        "picture.pairs[].lines[].points[].[]: number",
        "picture.pairs[].trace: any?",
        "picture.pairs[].curves: list",
        "picture.shown[]: str",
        "picture.center{}: number",
        "slice.x: str",
        "slice.y: str",
        "slice.at{}: number",
        "slice.region.kind: str",
        "slice.region.points[].[]: number",
        "slice.lines[].limit: int",
        "slice.lines[].points[].[]: number",
        "slice.trace: any?",
        "slice.curves: list",
    ],
    "curved picture": [
        "picture.pairs[].shadow.kind: str",
        "picture.pairs[].shadow.x[]: number",
        "picture.pairs[].shadow.y[]: number",
        "picture.pairs[].shadow.z[]: list",
        "picture.pairs[].trace.x[]: number",
        "picture.pairs[].trace.y[]: number",
        "picture.pairs[].trace.z[].[]: int",
        "picture.pairs[].trace.crossings[].[]: number",
        "picture.pairs[].curves[].limit: int",
        "picture.pairs[].curves[].trace.z[].[]: int",
        "picture.pairs[].curves[].trace.crossings[].[]: number",
    ],
    "curved slice": [
        "slice.region.kind: str",
        "slice.region.z[].[]: int",
        "slice.trace.x[]: number",
        "slice.trace.y[]: number",
        "slice.trace.z[].[]: int",
        "slice.trace.crossings[].[]: number",
        "slice.curves[].limit: int",
        "slice.curves[].trace.x[]: number",
        "slice.curves[].trace.crossings[].[]: number",
    ],
}

#: The entry point each reply comes from.
ENTRY_POINTS = {
    "catalogue": "api_catalogue",
    "design": "api_make_design",
    "simulation": "api_simulate",
    "analysis": "api_analyze_runs",
    "saturated analysis": "api_analyze_runs",
    "charts": "api_plots",
    "charts through the best settings": "api_plots",
    "best settings": "api_best_settings",
    "best settings on a limit": "api_best_settings",
    "confirmation runs judged": "api_best_settings",
    "best settings without a goal": "api_best_settings",
    "design of several responses": "api_make_design",
    "simulation of several responses": "api_simulate",
    "upload of several responses": "api_read_workbook",
    "relabel of several responses": "api_relabel",
    "workbook": "api_workbook",
    "relabel": "api_relabel",
    "upload": "api_read_workbook",
    "candidates": "api_candidates",
    "slow candidates": "api_candidates",
    "design with limits": "api_make_design",
    "limits": "api_limits",
    "limits with a problem": "api_limits",
    "limits proposed": "api_limits",
    "limits not proposed": "api_limits",
    "picture": "api_limits",
    "curved picture": "api_limits",
    "curved slice": "api_limits",
}

#: Entry points the page does not call: ``api_analyze`` analyses a workbook in one call, for the
#: tests (``tests/web/wasm`` among them) and for scripted use; the page reads a workbook and then
#: analyses its runs. ``api_compare_plan`` and ``api_compare`` are the two halves of
#: ``api_candidates``, which the page calls once for both.
NOT_CALLED_BY_THE_PAGE = {"api_analyze", "api_compare_plan", "api_compare"}

#: Three responses, as the brief lists them: a yield to maximise, a cost to minimise and a purity with a target.
THREE = [
    {"name": "Yield", "label": "Yield", "units": "%", "goal": "maximize"},
    {"name": "Cost", "label": "Cost", "units": "$/kg", "goal": "minimize"},
    {"name": "Purity", "label": "Purity", "units": "%", "goal": "target", "target": 63.0},
]

#: Two straight limits, one with a note: between them they narrow both ranges, so ``bounds`` has entries to read.
LIMITS = [{"expression": "T + 25*P <= 215", "note": "the product chars"}, {"expression": "400 <= 3*T + 5*P <= 600"}]

TYPES = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "bool": lambda v: isinstance(v, bool),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
    "any": lambda _: True,
}


def values_at(result: dict, path: str) -> list[Any]:
    """Return every value ``path`` reaches in ``result``, failing on a missing key or an empty list."""
    found: list[Any] = [result]
    for step in path.split("."):
        key, fan = re.fullmatch(r"(\w*)(\[\]|\{\})?", step).groups()
        if key:
            missing = [item for item in found if not isinstance(item, dict) or key not in item]
            assert not missing, f"{path}: no {key!r} in the reply"
            found = [item[key] for item in found]
        if fan:
            found = [x for item in found for x in (item if fan == "[]" else item.values())]
            assert found, f"{path}: nothing to check at {step!r}"
    return found


#: A Box-Behnken design measuring the three responses, as the page asks for it.
THREE_DESIGN = {"design_type": "box_behnken", "factors": FACTORS, "responses": THREE}


def several_responses(api) -> dict[str, Callable[[], dict]]:
    """Return the requests about a study of three responses: its design, its simulation, its workbook and a renaming."""

    def simulated() -> tuple[dict, dict]:
        design = held_of(api.ok("api_make_design", THREE_DESIGN))
        return design, api.ok("api_simulate", {**design, "seed": 2})

    def uploaded() -> dict:
        design, sim = simulated()
        results = {"responses": sim["responses"], "more_responses": {n: m["responses"] for n, m in sim["more"].items()}}
        marks = {"sources": ["simulated"] * 15, "more_sources": {n: ["simulated"] * 15 for n in sim["more"]}}
        sent = {**design, **results, **marks, "models": {"Cost": "interactions"}}
        return {"xlsx_base64": api.ok("api_workbook", sent)["xlsx_base64"]}

    def relabelled() -> dict:
        design, sim = simulated()
        return {"design": design, "factors": [{**f, "name": f"{f['name']}_new"} for f in FACTORS], "simulation": sim}

    return {
        "design of several responses": lambda: THREE_DESIGN,
        "simulation of several responses": lambda: {**simulated()[0], "seed": 2},
        "upload of several responses": uploaded,
        "relabel of several responses": relabelled,
    }


@pytest.fixture(scope="module")
def reply(api, worked_example: dict[str, Any]) -> Callable[[str], dict]:
    """Return the result of each kind of request, made once, when a test first asks for it.

    The requests reach every key the page reads.
    """
    held, simulated = worked_example["held"], worked_example["simulated"]
    analysed = {**held, "responses": simulated["responses"], "truth": simulated["truth"]}

    def saturated() -> dict:
        spec = {"design_type": "plackett_burman", "factors": FACTORS, "n_center_points": 0}
        return {**held_of(api.ok("api_make_design", spec)), "responses": [1.0, 2.5, 3.0, 5.5], "model": "main_effects"}

    def upload() -> dict:
        return api.ok("api_workbook", {**analysed, "sources": ["simulated"] * 15, "simulated": simulated})

    def slow() -> dict:
        return {"factors": [{"name": n, "low": -1, "high": 1} for n in "ABCDE"], "model": "quadratic"}

    def through_best() -> dict:
        best = api.ok("api_best_settings", analysed)
        return {**analysed, "best": best["coded"]}

    def on_a_limit() -> dict:
        spec = {"design_type": "d_optimal", "factors": FACTORS, "constraints": LIMITS}
        design = held_of(api.ok("api_make_design", {**spec, "response_info": {"goal": "maximize"}}))
        rows = [dict(zip(design["columns"], row, strict=True)) for row in design["rows"]]
        responses = [50 + 3 * (r["T"] - 175) / 25 + 2 * (r["P"] - 2) - ((r["F"] - 15) / 5) ** 2 for r in rows]
        return {**design, "responses": responses}

    def confirmed() -> dict:
        at = api.ok("api_best_settings", analysed)["confirm_at"]
        columns = [*held["columns"], "Confirmation run"]
        extra = [[{"Run order": 16 + i, "Confirmation run": "yes", **at}.get(c) for c in columns] for i in range(3)]
        rows = [*[[*row, None] for row in held["rows"]], *extra]
        return {**held, "columns": columns, "rows": rows, "responses": [*simulated["responses"], 67.0, 68.0, None]}

    def no_goal() -> dict:
        spec = {**held["spec"], "response_info": {**held["spec"]["response_info"], "goal": "learn"}}
        return {**analysed, "spec": spec}

    def relabel() -> dict:
        design = worked_example["design"]
        sent = {**held, "facts": design["facts"], "outside_range": design["outside_range"]}
        factors = [{**f, "name": f"{f['name']}_new"} for f in held["spec"]["factors"]]
        return {"design": sent, "factors": factors, "simulation": simulated}

    requests: dict[str, Callable[[], dict]] = {
        "catalogue": dict,
        "design": lambda: worked_example["demo"]["design"],
        "simulation": lambda: {**held, "seed": 2},
        "analysis": lambda: analysed,
        "saturated analysis": saturated,
        "charts": lambda: analysed,
        "charts through the best settings": through_best,
        "best settings": lambda: analysed,
        "best settings on a limit": on_a_limit,
        "confirmation runs judged": confirmed,
        "best settings without a goal": no_goal,
        **several_responses(api),
        "workbook": lambda: analysed,
        "relabel": relabel,
        "upload": upload,
        "candidates": lambda: {"factors": FACTORS, "model": "main_quadratic"},
        "slow candidates": slow,
        "design with limits": lambda: {"design_type": "d_optimal", "factors": FACTORS, "constraints": LIMITS},
        "limits": lambda: {"factors": FACTORS, "constraints": LIMITS},
        "limits with a problem": lambda: {"factors": FACTORS, "constraints": [{"expression": "T + Z <= 3"}]},
        "limits proposed": lambda: {
            "factors": FACTORS,
            "constraints": [],
            "marked": [6, 7],
            "cuts": {"marked": [6, 7], "corner": 6, "values": [180, 2.5, None]},
        },
        "limits not proposed": lambda: {
            "factors": FACTORS,
            "constraints": [],
            "marked": [8],
            "cuts": {"marked": [7], "corner": 7, "values": [200, None, None]},
        },
        "picture": lambda: {
            "factors": FACTORS,
            "constraints": [{"expression": "T + 25*P <= 250"}],
            "slice": {"x": "T", "y": "P", "at": {"F": 15}},
        },
        "curved picture": lambda: {"factors": FACTORS[:2], "constraints": [{"expression": "T * P <= 450"}]},
        "curved slice": lambda: {
            "factors": FACTORS,
            "constraints": [{"expression": "T * F <= 3000"}],
            "slice": {"x": "T", "y": "F", "at": {"P": 2}},
        },
    }
    return functools.cache(lambda kind: api.ok(ENTRY_POINTS[kind], requests[kind]()))


@pytest.mark.parametrize("kind", READS)
def test_each_reply_holds_what_the_page_reads(reply, kind):
    if kind.startswith("charts"):
        pytest.importorskip("plotly", reason="the page installs plotly for its first chart")
    for read in READS[kind]:
        path, expected = (part.strip() for part in read.split(":"))
        allowed, nullable = TYPES[expected.rstrip("?")], expected.endswith("?")
        wrong = [v for v in values_at(reply(kind), path) if not (allowed(v) or (nullable and v is None))]
        assert not wrong, f"{kind}: {path} should be {expected}, not {wrong[:3]}"


def test_each_chart_is_a_plotly_figure_the_page_can_restyle(reply):
    pytest.importorskip("plotly", reason="the page installs plotly for its first chart")
    figures = {key: json.loads(text) for key, text in reply("charts")["figures"].items()}
    for key, figure in figures.items():
        assert isinstance(figure["data"], list), key
        assert isinstance(figure.get("layout", {}), dict), key
    for key in ("contour", "true_contour"):  # the page gives these two one colour scale
        assert isinstance(figures[key]["data"][0]["z"], list), key


def test_each_measure_has_a_cell_for_each_design(reply):
    comparison = reply("candidates")["result"]
    for label in comparison["designs"]:
        assert len(comparison["display"][label]) == len(comparison["measures"]), label
        assert label in comparison["notes"], label


#: The page's modules that call Python.
CALLERS = ("app.js", "limits.js")


def test_the_page_calls_only_entry_points_this_contract_covers(bootstrap):
    called = {fn for module in CALLERS for fn in re.findall(r'call\("(api_\w+)"', source(module))}
    defined = {name for name in dir(bootstrap) if name.startswith("api_")}
    assert called <= defined, "app.js calls an entry point bootstrap.py does not define"
    assert called == set(ENTRY_POINTS.values()), "a new call from app.js needs its READS here"
    assert defined - called == NOT_CALLED_BY_THE_PAGE


def test_the_catalogue_is_what_bootstrap_offers(api, bootstrap):
    catalogue = api.ok("api_catalogue")
    assert catalogue["designs"] == bootstrap.DESIGNS
    assert catalogue["models"] == list(bootstrap.MODELS)
    assert catalogue["alphas"] == list(bootstrap.ALPHAS)
    assert catalogue["demo"] == bootstrap.DEMO
    assert list(catalogue["scenarios"]) == list(bootstrap.SCENARIOS)
    for key, scenario in bootstrap.SCENARIOS.items():
        offered = {name: scenario[name] for name in ("label", "hint", "sigma")}
        assert catalogue["scenarios"][key] == offered, key


def test_the_catalogue_names_the_release_a_notebook_installs(api):
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("process-improve")
    except PackageNotFoundError:
        installed = None
    assert api.ok("api_catalogue")["version"] == installed


def test_a_page_run_from_a_source_tree_names_no_release(api, bootstrap, monkeypatch):
    """Without an installed release to name, the notebook installs process-improve unpinned."""

    def not_installed(name: str) -> str:
        raise bootstrap.PackageNotFoundError(name)

    monkeypatch.setattr(bootstrap, "version", not_installed)
    assert api.ok("api_catalogue")["version"] is None
