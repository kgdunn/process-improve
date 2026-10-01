"""Tests for the no-install browser app (``web/``).

The app's Python side, ``web/bootstrap.py``, is ordinary Python, so it is tested
here natively. ``web/test-wasm.mjs`` repeats the round trip under real Pyodide in
CI; these tests are the fast half that runs with the rest of the suite.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("design_type", ["full_factorial", "plackett_burman", "dsd", "box_behnken", "ccd", "d_optimal"])
def test_every_offered_design_generates(app, design_type):
    reply = call(app, "api_make_design", {"design_type": design_type, "factors": FACTORS, "budget": 14})
    assert reply["ok"], reply.get("error")
    result = reply["result"]
    assert result["n_runs"] == len(result["rows"])
    assert result["columns"][:2] == ["Std order", "Run order"]
    run_order = [row[1] for row in result["rows"]]
    assert run_order == sorted(run_order), "rows are listed in the order they are to be run"


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
