"""Tests for the no-install browser app (``web/``).

The app's Python side, ``web/bootstrap.py``, is ordinary Python, so it is tested
here natively. ``web/test-wasm.mjs`` repeats the round trip under real Pyodide in
CI; these tests are the fast half that runs with the rest of the suite.
"""

from __future__ import annotations

import base64
import html.parser
import importlib.util
import io
import itertools
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import generate_design

pytest.importorskip("openpyxl")
pytest.importorskip("pyDOE3")

WEB = Path(__file__).resolve().parent.parent / "web"


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("web_bootstrap", WEB / "bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FACTORS = [
    {"name": "T", "low": 150, "high": 200, "units": "degC"},
    {"name": "P", "low": 1, "high": 3, "units": "bar"},
    {"name": "F", "low": 10, "high": 20, "units": ""},
]


def call(app, fn: str, payload: dict) -> dict:
    return json.loads(getattr(app, fn)(json.dumps(payload)))


def fill(xlsx_base64: str, response, blank_rows: tuple[int, ...] = ()) -> str:
    """Fill the Runs sheet's response column the way a reader would, in Excel."""
    sheets = pd.read_excel(io.BytesIO(base64.b64decode(xlsx_base64)), sheet_name=None)
    runs = sheets["Runs"]
    runs["y"] = response(runs)
    runs.loc[list(blank_rows), "y"] = np.nan
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xl:
        runs.to_excel(xl, sheet_name="Runs", index=False)
        sheets["_pi_design"].to_excel(xl, sheet_name="_pi_design", index=False)
    return base64.b64encode(buf.getvalue()).decode()


def quadratic(runs: pd.DataFrame) -> np.ndarray:
    t, p = (runs["T"] - 175) / 25, runs["P"] - 2
    noise = np.random.default_rng(0).normal(0, 0.1, len(runs))
    return (50 + 4 * t - 3 * p + 2 * t * p - 5 * t**2 + noise).to_numpy()


ALL_DESIGNS = [
    "full_factorial",
    "fractional_factorial",
    "plackett_burman",
    "dsd",
    "box_behnken",
    "ccd",
    "d_optimal",
    "i_optimal",
]


def run_sheet(result: dict) -> pd.DataFrame:
    """Return the run sheet a reply describes, as the reader sees it."""
    return pd.DataFrame(result["rows"], columns=result["columns"])


@pytest.mark.parametrize("design_type", ALL_DESIGNS)
def test_every_offered_design_generates(app, design_type):
    reply = call(app, "api_make_design", {"design_type": design_type, "factors": FACTORS, "budget": 14})
    assert reply["ok"], reply.get("error")
    result = reply["result"]
    assert result["n_runs"] == len(result["rows"])
    assert result["columns"][:2] == ["Std order", "Run order"]
    run_order = [row[1] for row in result["rows"]]
    assert run_order == sorted(run_order), "rows are listed in the order they are to be run"


@pytest.mark.parametrize("design_type", ALL_DESIGNS)
def test_run_sheet_lists_the_runs_in_their_randomised_order(app, design_type):
    """Row i of the sheet is the i-th run the library randomised, and "Std order" is where it sits unrandomised.

    ``DesignResult.design_actual`` is already in run order, and ``run_order[i]`` is the
    standard-order position of its i-th row. Sorting the sheet on that position instead
    lists the runs unrandomised, centre points last, under a "Run order" of 1..n.
    """
    spec = {"design_type": design_type, "factors": FACTORS, "budget": 14}
    sheet = run_sheet(call(app, "api_make_design", spec)["result"])
    design = generate_design(app._factors(spec), design_type=design_type, **app._design_options(spec, design_type))
    names = [f["name"] for f in FACTORS]
    np.testing.assert_allclose(sheet[names].to_numpy(float), design.design_actual[names].to_numpy(float))
    assert sheet["Run order"].tolist() == list(range(1, design.n_runs + 1))
    assert sheet["Std order"].tolist() == design.run_order


def coded(sheet: pd.DataFrame) -> np.ndarray:
    """Return the sheet's factor settings, coded to -1 and +1 from the ranges in ``FACTORS``."""
    return np.column_stack(
        [(sheet[f["name"]] - (f["high"] + f["low"]) / 2) / ((f["high"] - f["low"]) / 2) for f in FACTORS]
    )


def test_std_order_rebuilds_the_standard_order_factorial(app):
    """Sorted on "Std order", a 2^3 factorial reads corner by corner, last factor fastest, then its centre runs."""
    spec = {"design_type": "full_factorial", "factors": FACTORS, "n_center_points": 3}
    sheet = run_sheet(call(app, "api_make_design", spec)["result"])
    standard = sheet.sort_values("Std order")
    corners = np.array(list(itertools.product([-1, 1], repeat=3)))
    np.testing.assert_allclose(coded(standard), np.vstack([corners, np.zeros((3, 3))]))
    assert not np.allclose(coded(sheet), coded(standard)), "the runs are made in a randomised order"


def test_std_order_rebuilds_the_definitive_screening_foldover(app):
    """Sorted on "Std order", a DSD reads as its conference matrix C, then -C, then the centre run."""
    spec = {"design_type": "dsd", "factors": FACTORS}
    standard = coded(run_sheet(call(app, "api_make_design", spec)["result"]).sort_values("Std order"))
    half = (len(standard) - 1) // 2
    np.testing.assert_allclose(standard[:half], -standard[half : 2 * half])
    np.testing.assert_allclose(standard[-1], 0)


def factors_named(*names: str) -> list[dict]:
    """Return coded-range factors with the given names."""
    return [{"name": n, "low": -1, "high": 1, "units": ""} for n in names]


SIX = factors_named("A", "B", "C", "D", "E", "G")


@pytest.mark.parametrize("design_type", ["d_optimal", "i_optimal"])
def test_optimal_design_leaves_degrees_of_freedom_for_error_by_default(app, design_type):
    """With no budget given, an optimal design has four runs more than its model has coefficients.

    The page used to send 16 runs whatever the model: six factors and the quadratic
    model have 28 coefficients, so the library raised the budget to exactly 28 and the
    fit had nothing left to estimate the error with.
    """
    spec = {"design_type": design_type, "factors": SIX[:5], "model_type": "quadratic"}
    result = call(app, "api_make_design", spec)["result"]
    assert result["n_runs"] == 21 + 4


def test_a_raised_budget_is_reported(app):
    spec = {"design_type": "d_optimal", "factors": SIX, "model_type": "quadratic", "budget": 16}
    result = call(app, "api_make_design", spec)["result"]
    assert result["n_runs"] == 28
    assert any("16 runs" in note and "28 coefficients" in note for note in result["facts"]["notes"])


@pytest.mark.parametrize(("design_type", "extra"), [("box_behnken", {}), ("ccd", {"alpha": "rotatable"})])
def test_designs_singular_without_a_center_point_need_one(app, design_type, extra):
    """Without a center run, every other run of these designs lies on one sphere, so the squares are aliased."""
    spec = {"design_type": design_type, "factors": FACTORS, "n_center_points": 0, **extra}
    reply = call(app, "api_make_design", spec)
    assert not reply["ok"]
    assert "at least one center point" in reply["error"]


def test_face_centred_ccd_may_skip_center_points(app):
    spec = {"design_type": "ccd", "factors": FACTORS, "n_center_points": 0, "alpha": "face_centered"}
    assert call(app, "api_make_design", spec)["result"]["n_runs"] == 8 + 6


@pytest.mark.parametrize(
    ("k", "resolution", "n_runs", "achieved"),
    [(3, 3, 4, 3), (4, 4, 8, 4), (5, 4, 16, 5), (6, 3, 8, 3), (6, 4, 16, 4), (6, 5, 32, 6), (7, 4, 16, 4)],
)
def test_fractional_factorial_takes_the_resolution_asked_for(app, k, resolution, n_runs, achieved):
    spec = {
        "design_type": "fractional_factorial",
        "factors": SIX[:k] if k <= 6 else factors_named(*"ABCDEFG"),
        "resolution": resolution,
        "n_center_points": 0,
    }
    result = call(app, "api_make_design", spec)["result"]
    assert result["n_runs"] == n_runs
    assert result["facts"]["resolution"] == achieved


def test_fractional_factorial_defaults_to_resolution_iv_not_the_half_fraction(app):
    """Six factors: 16 runs at resolution IV, where the half fraction the page used to send has 32."""
    spec = {"design_type": "fractional_factorial", "factors": SIX, "n_center_points": 0}
    assert call(app, "api_make_design", spec)["result"]["n_runs"] == 16


def test_fractional_factorial_refuses_a_resolution_above_the_factor_count(app):
    reply = call(app, "api_make_design", {"design_type": "fractional_factorial", "factors": FACTORS, "resolution": 4})
    assert not reply["ok"]
    assert "resolution III" in reply["error"]


@pytest.mark.parametrize(("k", "model"), [(4, "main_effects"), (5, "interactions")])
def test_fraction_is_analysed_with_the_model_its_resolution_supports(app, k, model):
    """Resolution IV keeps main effects clear; resolution V keeps two-factor interactions clear too."""
    spec = {"design_type": "fractional_factorial", "factors": SIX[:k], "resolution": 4}
    assert call(app, "api_make_design", spec)["result"]["model"] == model


def test_ccd_uses_a_half_fraction_cube_from_six_factors(app):
    spec = {"design_type": "ccd", "factors": SIX, "n_center_points": 3, "alpha": "face_centered"}
    assert call(app, "api_make_design", spec)["result"]["n_runs"] == 32 + 12 + 3


@pytest.mark.parametrize(
    ("design_type", "k", "word"),
    [("dsd", 2, "3 to"), ("box_behnken", 2, "3 to"), ("fractional_factorial", 2, "3 to"), ("i_optimal", 7, "to 6")],
)
def test_designs_outside_their_factor_range_are_refused(app, design_type, k, word):
    reply = call(app, "api_make_design", {"design_type": design_type, "factors": factors_named(*"ABCDEFG"[:k])})
    assert not reply["ok"]
    assert word in reply["error"]


def test_facts_name_main_effects_aliased_with_interactions(app):
    """A 2^(3-1) design: each main effect shares its column with the other two factors' interaction."""
    spec = {"design_type": "fractional_factorial", "factors": FACTORS}
    facts = call(app, "api_make_design", spec)["result"]["facts"]
    assert sorted(sorted(group) for group in facts["aliases"]) == [["F", "T·P"], ["P", "T·F"], ["P·F", "T"]]
    assert any("Main effects are aliased with two-factor interactions" in note for note in facts["notes"])


def test_facts_name_interactions_aliased_with_each_other(app):
    names = ("Temp", "Pres", "Flow", "Cat")
    spec = {"design_type": "fractional_factorial", "factors": factors_named(*names), "resolution": 4}
    facts = call(app, "api_make_design", spec)["result"]["facts"]
    assert sorted(sorted(group) for group in facts["aliases"]) == [
        ["Flow·Cat", "Temp·Pres"],
        ["Pres·Cat", "Temp·Flow"],
        ["Pres·Flow", "Temp·Cat"],
    ]
    assert facts["generators"] == ["Cat = Temp·Pres·Flow"]


@pytest.mark.parametrize("design_type", ["full_factorial", "dsd", "box_behnken", "ccd"])
def test_facts_say_when_nothing_is_fully_aliased(app, design_type):
    facts = call(app, "api_make_design", {"design_type": design_type, "factors": FACTORS})["result"]["facts"]
    assert facts["aliases"] == []
    assert any("No main effect or two-factor interaction is fully aliased" in note for note in facts["notes"])


def test_plackett_burman_center_points_can_be_chosen(app):
    spec = {"design_type": "plackett_burman", "factors": FACTORS, "n_center_points": 0}
    result = call(app, "api_make_design", spec)["result"]
    assert result["n_runs"] == 4
    assert result["facts"]["center_runs"] == 0


def test_workbook_round_trip_recovers_the_model(app):
    design = call(app, "api_make_design", {"design_type": "box_behnken", "factors": FACTORS})["result"]
    reply = call(app, "api_analyze", {"xlsx_base64": fill(design["xlsx_base64"], quadratic)})
    assert reply["ok"], reply.get("error")
    coef = {c["term"]: c["coefficient"] for c in reply["result"]["coefficients"]}
    assert coef["T"] == pytest.approx(4, abs=0.2)
    assert coef["T:P"] == pytest.approx(2, abs=0.2)
    assert coef["I(T ** 2)"] == pytest.approx(-5, abs=0.3)
    assert reply["result"]["equation"].startswith("y = ")


def test_partly_filled_workbook_reports_pending_runs(app):
    design = call(app, "api_make_design", {"design_type": "box_behnken", "factors": FACTORS})["result"]
    xlsx = fill(design["xlsx_base64"], quadratic, blank_rows=(0, 4))
    result = call(app, "api_analyze", {"xlsx_base64": xlsx, "model": "interactions"})["result"]
    assert result["pending_runs"] == [1, 5]
    assert result["n_completed"] == design["n_runs"] - 2


def test_too_few_runs_says_how_many_are_needed(app):
    design = call(app, "api_make_design", {"design_type": "box_behnken", "factors": FACTORS})["result"]
    xlsx = fill(design["xlsx_base64"], quadratic, blank_rows=tuple(range(8)))
    reply = call(app, "api_analyze", {"xlsx_base64": xlsx, "model": "quadratic"})
    assert not reply["ok"]
    assert "10 coefficients" in reply["error"]


def test_face_centred_ccd_stays_inside_the_ranges(app):
    face = call(app, "api_make_design", {"design_type": "ccd", "factors": FACTORS, "alpha": "face_centered"})
    rotatable = call(app, "api_make_design", {"design_type": "ccd", "factors": FACTORS, "alpha": "rotatable"})
    assert face["result"]["outside_range"] == []
    assert rotatable["result"]["outside_range"] == ["T", "P", "F"]


@pytest.mark.parametrize("model", ["__import__('os').system('true')", "y ~ T", ""])
def test_model_name_from_the_request_is_never_a_formula(app, model):
    design = call(app, "api_make_design", {"design_type": "box_behnken", "factors": FACTORS})["result"]
    reply = call(app, "api_analyze", {"xlsx_base64": fill(design["xlsx_base64"], quadratic), "model": model})
    assert reply["ok"] is (model == ""), "an empty choice means 'as designed'; anything else not in MODELS is refused"


@pytest.mark.parametrize(
    ("factors", "message"),
    [
        (FACTORS[:1], "at least two"),
        ([FACTORS[0], {**FACTORS[1], "name": "T"}], "unique"),
        ([FACTORS[0], {**FACTORS[1], "name": "a b"}], "variable names"),
        ([FACTORS[0], {**FACTORS[1], "low": 5}], "below the high"),
    ],
)
def test_bad_factor_input_is_reported(app, factors, message):
    reply = call(app, "api_make_design", {"design_type": "box_behnken", "factors": factors})
    assert not reply["ok"]
    assert message in reply["error"]


class _DesignOptions(html.parser.HTMLParser):
    """Collect the <option>s of the page's design-type <select>."""

    def __init__(self) -> None:
        super().__init__()
        self.inside = False
        self.options: list[dict] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select":
            self.inside = attrs.get("id") == "design-type"
        elif tag == "option" and self.inside:
            self.options.append({**attrs, "label": ""})

    def handle_endtag(self, tag):
        if tag == "select":
            self.inside = False

    def handle_data(self, data):
        if self.inside and self.options:
            self.options[-1]["label"] += data.strip()


def test_page_design_list_matches_bootstrap(app):
    """The page lists the designs in HTML so the form is right before Python loads.

    That copy must say what ``bootstrap.DESIGNS`` says, or a reader would see one
    design's settings and generate another's.
    """
    parser = _DesignOptions()
    parser.feed((WEB / "index.html").read_text(encoding="utf-8"))
    in_page = {o["value"]: o for o in parser.options}
    assert list(in_page) == list(app.DESIGNS)
    for key, info in app.DESIGNS.items():
        option = in_page[key]
        assert option["label"] == info["label"], key
        assert option["data-hint"] == info["hint"], key
        assert option["data-options"].split() == info["options"], key
        assert int(option["data-min-factors"]) == info["min_factors"], key
        assert int(option["data-max-factors"]) == info["max_factors"], key
    selected = [o["value"] for o in parser.options if "selected" in o]
    assert selected == ["dsd"], "the page opens with definitive screening selected"


def test_foreign_file_is_reported(app):
    reply = call(app, "api_analyze", {"xlsx_base64": base64.b64encode(b"not a workbook").decode()})
    assert not reply["ok"]
    assert "xlsx" in reply["error"]


@pytest.mark.slow
def test_app_imports_without_native_only_extras():
    """The browser has no WebAssembly build of these; the app must not need them."""
    script = textwrap.dedent(
        f"""
        import importlib.abc, sys

        BLOCKED = {{"pulp", "pyoptex", "numba", "osqp", "skimage", "ruptures", "mcp", "matplotlib", "plotly"}}

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"{{name}} is not available in the browser")

        sys.meta_path.insert(0, Block())
        sys.path.insert(0, {str(WEB)!r})
        import json, bootstrap
        factors = [{{"name": n, "low": 0, "high": 1}} for n in "ABC"]
        for design in bootstrap.DESIGNS:
            reply = json.loads(bootstrap.api_make_design(json.dumps({{"design_type": design, "factors": factors}})))
            assert reply["ok"], (design, reply)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr
