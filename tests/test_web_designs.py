"""Every design the browser app offers, checked through the page's own code path.

``web/bootstrap.py`` builds the designs the page shows and analyses the runs a reader
fills in. These tests drive it as the page does, over the factor counts and options
each design takes, and check what makes each design the design it claims to be,
rather than pinned numbers:

- the run sheet: settings inside the ranges entered (or at the stated axial distance),
  a randomised permutation of the standard order, the same sheet for the same seed;
- the design's defining structure: corners, resolution and generators, fold-over,
  orthogonality, axial distance, published run counts, full rank;
- a round trip: a known model in the design's own terms plus small noise, written into
  the downloaded workbook and analysed as uploaded, recovers the coefficients, and the
  coefficients, standard errors, R^2, adjusted and predicted R^2, residual sum of squares
  and lack-of-fit F equal an independent numpy least-squares fit.

The optimal designs use the built-in exchange, the code path the browser runs.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import moment_aberration

pytest.importorskip("openpyxl")

WEB = Path(__file__).resolve().parent.parent / "web"

#: Factor names: single letters, avoiding "I", which a defining relation uses for the identity.
NAMES = ("T", "P", "F", "C", "S", "R", "M", "W", "V", "Q", "N", "L")

#: Box-Behnken run counts without center runs (Box and Behnken, 1960; NIST/SEMATECH).
BOX_BEHNKEN_RUNS = {3: 12, 4: 24, 5: 40, 6: 48, 7: 56}

#: Noise added to the known model in the round trip.
SIGMA = 0.01


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("web_bootstrap_designs", WEB / "bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call(app, fn: str, payload: dict) -> dict:
    reply = json.loads(getattr(app, fn)(json.dumps(payload)))
    assert reply["ok"], reply.get("error")
    return reply["result"]


def factors(k: int) -> list[dict]:
    """Return k factors in real units, each with its own range, so the coding is exercised."""
    return [{"name": NAMES[i], "low": 100 + 10 * i, "high": 100 + 10 * i + 2 * (i + 1), "units": ""} for i in range(k)]


def coded(sheet: pd.DataFrame, spec_factors: list[dict]) -> np.ndarray:
    """Return the sheet's settings coded to -1 and +1 from the ranges entered."""
    return np.column_stack(
        [(sheet[f["name"]] - (f["high"] + f["low"]) / 2) / ((f["high"] - f["low"]) / 2) for f in spec_factors]
    )


# --------------------------------------------------------------------------- model terms
def model_terms(k: int, model: str) -> list[tuple[int, ...]]:
    """List a named model's terms as tuples of factor indices; ``(i, i)`` is a square."""
    terms: list[tuple[int, ...]] = [(), *[(i,) for i in range(k)]]
    if model in {"interactions", "quadratic"}:
        terms += list(itertools.combinations(range(k), 2))
    if model == "quadratic":
        terms += [(i, i) for i in range(k)]
    return terms


def term_name(term: tuple[int, ...]) -> str:
    """Name a term as ``analyze_experiment`` reports it."""
    if not term:
        return "Intercept"
    if len(term) == 2 and term[0] == term[1]:
        return f"I({NAMES[term[0]]} ** 2)"
    return ":".join(NAMES[i] for i in term)


def model_matrix(x: np.ndarray, terms: list[tuple[int, ...]]) -> np.ndarray:
    """Expand coded runs into the model's columns."""
    return np.column_stack([np.prod(x[:, list(t)], axis=1) if t else np.ones(len(x)) for t in terms])


def known_coefficients(terms: list[tuple[int, ...]]) -> np.ndarray:
    """Give every term a distinct, non-zero coefficient."""
    values = {0: 10.0, 1: 1.0, 2: 0.4}
    return np.array([values[len(set(t))] * (1 + 0.15 * sum(t)) * (-1) ** sum(t) if t else 10.0 for t in terms])


# --------------------------------------------------------------------------- the cases
def _optimal(top: int) -> list[tuple[int, dict]]:
    """Budgets at, just above and at twice the coefficient count; one budget from five factors."""
    grid = []
    for k, model in itertools.product(range(2, top + 1), ("main_effects", "interactions", "quadratic")):
        p = len(model_terms(k, model))
        grid += [(k, {"model_type": model, "budget": b}) for b in ((p, p + 4, 2 * p) if k <= 4 else (p + 4,))]
    return grid


#: Every design type over the factor counts and options the page offers it.
GRID: dict[str, list[tuple[int, dict]]] = {
    "full_factorial": [(k, {"n_center_points": cp}) for k in range(2, 7) for cp in (0, 3)],
    "fractional_factorial": [
        *[(k, {"resolution": r, "n_center_points": 3}) for k in range(3, 9) for r in (3, 4, 5) if r <= k],
        (5, {"resolution": 4, "n_center_points": 0}),
    ],
    "plackett_burman": [
        *[(k, {"n_center_points": 3}) for k in range(2, 12)],
        *[(k, {"n_center_points": 0}) for k in (3, 7, 11)],
    ],
    "dsd": [(k, {}) for k in range(3, 11)],
    "box_behnken": [(k, {"n_center_points": cp}) for k in range(3, 8) for cp in (1, 3)],
    "ccd": [
        *[
            (k, {"alpha": a, "n_center_points": 3})
            for k in range(2, 8)
            for a in ("face_centered", "rotatable", "orthogonal")
        ],
        (3, {"alpha": "face_centered", "n_center_points": 0}),
    ],
    "d_optimal": _optimal(8),
    "i_optimal": _optimal(6),
}

CASES = [
    pytest.param(
        design_type,
        k,
        options,
        id="-".join([design_type, f"k{k}", *(f"{key}={value}" for key, value in options.items())]),
        marks=[pytest.mark.slow] if design_type.endswith("_optimal") and k >= 5 else [],
    )
    for design_type, grid in GRID.items()
    for k, options in grid
]


# --------------------------------------------------------------------------- the run sheet
@pytest.mark.parametrize(("design_type", "k", "options"), CASES)
def test_design(app, design_type: str, k: int, options: dict) -> None:
    """The run sheet, the design's defining structure, and the round trip, for one case."""
    spec = {"design_type": design_type, "factors": factors(k), **options}
    result = call(app, "api_make_design", spec)
    sheet = pd.DataFrame(result["rows"], columns=result["columns"])
    x = coded(sheet, spec["factors"])
    n = len(sheet)

    # The sheet: the runs in a randomised order, numbered 1..n, each with its standard-order place.
    assert sheet["Run order"].tolist() == list(range(1, n + 1))
    assert sorted(sheet["Std order"]) == list(range(1, n + 1))
    if n >= 6:
        assert sheet["Std order"].tolist() != list(range(1, n + 1)), "the runs are randomised"
    # The same seed gives the same sheet.
    assert call(app, "api_make_design", spec)["rows"] == result["rows"]
    # The facts the page states match the runs.
    assert result["facts"]["center_runs"] == int(np.all(np.isclose(x, 0), axis=1).sum())

    STRUCTURE[design_type](x, k, options, result)
    _round_trip(app, result, spec, x)


# --------------------------------------------------------------------------- structure
def _two_level_core(x: np.ndarray) -> np.ndarray:
    """Return the runs that are not center runs, checking they sit at -1 and +1 only."""
    core = x[~np.all(np.isclose(x, 0), axis=1)]
    np.testing.assert_allclose(np.abs(core), 1.0, atol=1e-9)
    return core


def _check_full_factorial(x: np.ndarray, k: int, options: dict, _result: dict) -> None:
    core = _two_level_core(x)
    corners = sorted(map(tuple, np.round(core).astype(int)))
    assert corners == sorted(itertools.product([-1, 1], repeat=k)), "each of the 2^k corners once"
    assert len(x) == 2**k + options["n_center_points"]
    gram = model_matrix(x, model_terms(k, "interactions"))
    gram = gram.T @ gram
    np.testing.assert_allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-9)


def _check_fractional_factorial(x: np.ndarray, k: int, options: dict, result: dict) -> None:
    core = _two_level_core(x)
    assert len(core) & (len(core) - 1) == 0, "a regular fraction has a power-of-two number of runs"
    assert len(core) < 2**k
    measured = moment_aberration(pd.DataFrame(core, columns=list(NAMES[:k]))).resolution
    assert result["facts"]["resolution"] == measured >= options["resolution"]
    for generator in result["facts"]["generators"]:
        left, right = generator.split(" = ")
        sign = -1.0 if right.startswith("-") else 1.0
        product = np.prod(core[:, [NAMES.index(f) for f in right.lstrip("-").split("·")]], axis=1)
        np.testing.assert_allclose(core[:, NAMES.index(left)], sign * product)


def _check_plackett_burman(x: np.ndarray, k: int, options: dict, _result: dict) -> None:
    core = _two_level_core(x)
    assert len(core) == 4 * (k // 4 + 1)
    assert len(x) == len(core) + options["n_center_points"]
    np.testing.assert_allclose(core.sum(axis=0), 0, atol=1e-9)
    gram = core.T @ core
    np.testing.assert_allclose(gram, len(core) * np.eye(k), atol=1e-9)


def _check_dsd(x: np.ndarray, k: int, _options: dict, _result: dict) -> None:
    centre = np.all(np.isclose(x, 0), axis=1)
    assert centre.sum() == 1
    folded = x[~centre]
    assert len(x) % 2 == 1
    assert (len(x) - 1) // 2 >= k
    runs = {tuple(np.round(r, 9)) for r in folded}
    assert all(tuple(np.round(-r, 9)) in runs for r in folded), "every run has its mirror image"
    np.testing.assert_allclose(sorted(set(np.round(x.ravel(), 9))), [-1, 0, 1])
    second = model_matrix(x, [t for t in model_terms(k, "quadratic") if len(t) == 2])
    np.testing.assert_allclose(x.T @ second, 0, atol=1e-9)
    gram = x.T @ x
    np.testing.assert_allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-9)


def _check_box_behnken(x: np.ndarray, k: int, options: dict, _result: dict) -> None:
    assert len(x) == BOX_BEHNKEN_RUNS[k] + options["n_center_points"]
    np.testing.assert_allclose(sorted(set(np.round(x.ravel(), 9))), [-1, 0, 1])
    assert not np.any(np.all(np.isclose(np.abs(x), 1), axis=1)), "no run at a corner"
    quadratic = model_matrix(x, model_terms(k, "quadratic"))
    assert np.linalg.matrix_rank(quadratic) == quadratic.shape[1]


def _check_ccd(x: np.ndarray, k: int, options: dict, _result: dict) -> None:
    cube = 2**k if k < 6 else 2 ** (k - 1)
    assert len(x) == cube + 2 * k + options["n_center_points"]
    axial = x[np.sum(~np.isclose(x, 0), axis=1) == 1]
    assert len(axial) == 2 * k
    alpha = np.abs(axial).max()
    np.testing.assert_allclose(np.abs(axial).sum(axis=1), alpha)
    if options["alpha"] == "face_centered":
        assert alpha == pytest.approx(1.0)
    elif options["alpha"] == "rotatable":
        assert alpha == pytest.approx(cube**0.25)
    else:
        squares = x**2 - (x**2).mean(axis=0)
        gram = squares.T @ squares
        np.testing.assert_allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-8 * gram.max())


def _check_optimal(x: np.ndarray, k: int, options: dict, _result: dict) -> None:
    terms = model_terms(k, options["model_type"])
    assert len(x) == max(options["budget"], len(terms))
    matrix = model_matrix(x, terms)
    assert np.linalg.matrix_rank(matrix) == len(terms)
    if options["model_type"] == "main_effects" and len(x) % 4 == 0:
        d_efficiency = np.linalg.det(matrix.T @ matrix) ** (1 / len(terms)) / len(x)
        assert d_efficiency == pytest.approx(1.0), "an orthogonal design exists at this size"


STRUCTURE = {
    "full_factorial": _check_full_factorial,
    "fractional_factorial": _check_fractional_factorial,
    "plackett_burman": _check_plackett_burman,
    "dsd": _check_dsd,
    "box_behnken": _check_box_behnken,
    "ccd": _check_ccd,
    "d_optimal": _check_optimal,
    "i_optimal": _check_optimal,
}


# --------------------------------------------------------------------------- the round trip
def _round_trip(app, result: dict, spec: dict, x: np.ndarray) -> None:
    """Fill the workbook with a known model plus small noise, analyse it as uploaded, and check the fit."""
    k = x.shape[1]
    terms = model_terms(k, result["model"])
    matrix = model_matrix(x, terms)
    beta = known_coefficients(terms)
    y = matrix @ beta + np.random.default_rng(len(x) + 7 * k).normal(0, SIGMA, len(x))

    sheets = pd.read_excel(io.BytesIO(base64.b64decode(result["xlsx_base64"])), sheet_name=None)
    sheets["Runs"]["y"] = y
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xl:
        for name, frame in sheets.items():
            frame.to_excel(xl, sheet_name=name, index=False)
    analysis = call(app, "api_analyze", {"xlsx_base64": base64.b64encode(buf.getvalue()).decode()})

    got = {c["term"]: c for c in analysis["coefficients"]}
    assert set(got) == {term_name(t) for t in terms}, "the design fits its own model without aliasing"
    estimate, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    for term, b, true in zip(terms, estimate, beta, strict=True):
        assert got[term_name(term)]["coefficient"] == pytest.approx(b, rel=1e-6, abs=1e-9)
        assert got[term_name(term)]["coefficient"] == pytest.approx(true, abs=0.05)

    n, p = matrix.shape
    residual = y - matrix @ estimate
    sse = float(residual @ residual)
    sst = float(((y - y.mean()) ** 2).sum())
    summary = analysis["summary"]
    assert summary["df_residual"] == n - p
    assert summary["r_squared"] == pytest.approx(1 - sse / sst, abs=1e-9)
    if n == p:
        assert analysis["lenth"] is not None or any("no degrees of freedom" in note for note in analysis["notes"])
        return

    inverse = np.linalg.inv(matrix.T @ matrix)
    standard_errors = np.sqrt(sse / (n - p) * np.diag(inverse))
    for term, se in zip(terms, standard_errors, strict=True):
        assert got[term_name(term)]["std_error"] == pytest.approx(se, rel=1e-6)
    assert summary["r_squared_adj"] == pytest.approx(1 - (sse / (n - p)) / (sst / (n - 1)), abs=1e-9)
    leverage = np.einsum("ij,jk,ik->i", matrix, inverse, matrix)
    if np.all(leverage < 1 - 1e-9):
        press = float(((residual / (1 - leverage)) ** 2).sum())
        assert summary["r_squared_pred"] == pytest.approx(1 - press / sst, abs=1e-6)
    residual_row = next(row for row in analysis["anova"] if row["source"] == "Residual")
    assert residual_row["sum_sq"] == pytest.approx(sse, rel=1e-6, abs=1e-12)
    assert residual_row["df"] == n - p
    _check_lack_of_fit(analysis["lack_of_fit"], x, y, sse, n - p)


def _check_lack_of_fit(lack_of_fit: dict | None, x: np.ndarray, y: np.ndarray, sse: float, df_residual: int) -> None:
    """Pure error comes from runs repeated at the same settings; the rest of the residual is lack of fit."""
    settings = pd.Series([" ".join(f"{v:.9g}" for v in row) for row in np.round(x, 9)])
    groups = pd.Series(y).groupby(settings)
    pure_error = float(groups.apply(lambda g: ((g - g.mean()) ** 2).sum()).sum())
    df_pure = int((groups.size() - 1).sum())
    df_lack = df_residual - df_pure
    if df_pure == 0 or df_lack == 0:
        return
    f_statistic = ((sse - pure_error) / df_lack) / (pure_error / df_pure)
    assert lack_of_fit["f_statistic"] == pytest.approx(f_statistic, rel=1e-6)
